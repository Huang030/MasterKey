import torch
import numpy as np
import hdbscan
import copy

def vectorize_net(net):
    return torch.cat([p.view(-1) for p in net.parameters()])

def load_model_weight(net, weight):
    index_bias = 0
    for p_index, p in enumerate(net.parameters()):
        p.data =  weight[index_bias:index_bias+p.numel()].view(p.size())
        index_bias += p.numel()

def load_model_weight_diff(net, weight_diff, global_weight):
    """
    load rule: w_t + clipped(w^{local}_t - w_t)
    """
    listed_global_weight = list(global_weight.parameters())
    index_bias = 0
    for p_index, p in enumerate(net.parameters()):
        p.data =  weight_diff[index_bias:index_bias+p.numel()].view(p.size()) + listed_global_weight[p_index]
        index_bias += p.numel()
        
class Defense:
    def __init__(self, *args, **kwargs):
        self.hyper_params = None

    def exec(self, client_model, *args, **kwargs):
        raise NotImplementedError()


class AddNoise(Defense):
    def __init__(self, stddev, *args, **kwargs):
        self.stddev = stddev

    def exec(self, client_model, *args, **kwargs):
        vectorized_net = vectorize_net(client_model)
        gaussian_noise = torch.randn(vectorized_net.size()).cuda() * self.stddev
        dp_weight = vectorized_net + gaussian_noise
        load_model_weight(client_model, dp_weight)
        print ("Weak DP Defense: added noise of norm: {}".format(torch.norm(gaussian_noise)))
        
        return None
    

class WeightDiffClippingDefense(Defense):
    def __init__(self, norm_bound, *args, **kwargs):
        self.norm_bound = norm_bound

    def exec(self, client_model, global_model, *args, **kwargs):
        """
        global_model: the global model at iteration T, bcast from the PS
        client_model: starting from `global_model`, the model on the clients after local retraining
        """
        vectorized_client_net = vectorize_net(client_model)
        vectorized_global_net = vectorize_net(global_model)
        vectorize_diff = vectorized_client_net - vectorized_global_net

        weight_diff_norm = torch.norm(vectorize_diff).item()
        clipped_weight_diff = vectorize_diff/max(1, weight_diff_norm/self.norm_bound)

        print ("Norm Weight Diff: {}, Norm Clipped Weight Diff {}".format(weight_diff_norm,
            torch.norm(clipped_weight_diff).item()))
        load_model_weight_diff(client_model, clipped_weight_diff, global_model)
        return None


class Krum(Defense):
    """
    we implement the robust aggregator at: https://papers.nips.cc/paper/6617-machine-learning-with-adversaries-byzantine-tolerant-gradient-descent.pdf
    and we integrate both krum and multi-krum in this single class
    """
    def __init__(self, mode, num_workers, num_adv, *args, **kwargs):
        assert (mode in ("krum", "multi-krum"))
        self._mode = mode
        self.num_workers = num_workers
        self.s = num_adv

    def exec(self, client_models, num_dps, *args, **kwargs):
        vectorize_nets = [vectorize_net(cm).detach() for cm in client_models]

        neighbor_distances = []
        for i, g_i in enumerate(vectorize_nets):
            distance = []
            for j in range(i+1, len(vectorize_nets)):
                if i != j:
                    g_j = vectorize_nets[j]
                    distance.append(torch.norm(g_i - g_j).pow(2).item())
            neighbor_distances.append(distance)

        # compute scores
        nb_in_score = self.num_workers - self.s - 2
        scores = []
        for i, g_i in enumerate(vectorize_nets):
            dists = []
            for j, g_j in enumerate(vectorize_nets):
                if j == i:
                    continue
                if j < i:
                    dists.append(neighbor_distances[j][i - j - 1])
                else:
                    dists.append(neighbor_distances[i][j - i - 1])
            # alternative to topk in PyTorch
            dists_tensor = torch.tensor(dists)
            topk_values, _ = torch.topk(dists_tensor, nb_in_score)
            scores.append(torch.sum(topk_values).item())
        if self._mode == "krum":
            i_star = scores.index(min(scores))
            aggregated_model = client_models[0]  # create a clone of the model
            aggregated_model.load_state_dict(client_models[i_star].state_dict())
            neo_net_list = [aggregated_model]
            neo_net_freq = [1.0]
            return neo_net_list, neo_net_freq
        
        elif self._mode == "multi-krum":
            topk_ind = np.argpartition(scores, nb_in_score+2)[:nb_in_score+2]

            # We reconstruct the weighted averaging here:
            selected_num_dps = np.array(num_dps)[topk_ind]
            reconstructed_freq = torch.tensor([snd/sum(selected_num_dps) for snd in selected_num_dps], dtype=torch.float32).cuda()

            aggregated_grad = torch.sum(torch.stack([reconstructed_freq[i] * vectorize_nets[j] for i, j in enumerate(topk_ind)], dim=0), dim=0)  # Weighted sum of the gradients
            
            aggregated_model = client_models[0]  # create a clone of the model
            load_model_weight(aggregated_model, aggregated_grad)
            neo_net_list = [aggregated_model]
            neo_net_freq = [1.0]
            return neo_net_list, neo_net_freq
        

class RLR(Defense):
    def __init__(self, n_params, args, robustLR_threshold = 0):
        self.args = args
        self.n_params = n_params
        self.robustLR_threshold = robustLR_threshold
        
         
    def exec(self, global_model, client_models, num_dps):
        # adjust LR if robust LR is selected
        print(f"self.args: {self.args}")
        print(f"self.args['server_lr']: {self.args['server_lr']}")
        lr_vector = torch.Tensor([self.args['server_lr']]*self.n_params).cuda()
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]
        vectorize_avg_net = vectorize_net(global_model).detach().cpu().numpy()
        local_updates = vectorize_nets - vectorize_avg_net
        aggr_freq = [num_dp/sum(num_dps) for num_dp in num_dps]
        
        if self.robustLR_threshold > 0:
            lr_vector = self.compute_robustLR(local_updates)
        
        
        aggregated_updates = 0
        if self.args['aggr']=='avg':          
            aggregated_updates = self.agg_avg(local_updates, aggr_freq)
        elif self.args['aggr']=='comed':
            #TODO update for the 2 remaining func
            aggregated_updates = self.agg_comed(local_updates)
        elif self.args['aggr'] == 'sign':
            aggregated_updates = self.agg_sign(local_updates)
            
        if self.args['noise'] > 0:
            aggregated_updates.add_(torch.normal(mean=0, std=self.args['noise']*self.args['clip'], size=(self.n_params,)).cuda())

        cur_global_params = vectorize_avg_net
        new_global_params =  (cur_global_params + lr_vector*aggregated_updates).astype(np.float32)
        
        aggregated_model = client_models[0] # slicing which doesn't really matter
        load_model_weight(aggregated_model, torch.from_numpy(new_global_params).cuda())
        neo_net_list = [aggregated_model]
        neo_net_freq = [1.0]
        return neo_net_list, neo_net_freq
        
    def compute_robustLR(self, agent_updates):
        agent_updates_sign = [np.sign(update) for update in agent_updates]  
        sm_of_signs = np.abs(sum(agent_updates_sign))
        print(f"sm_of_signs is: {sm_of_signs}")
        
        sm_of_signs[sm_of_signs < self.robustLR_threshold] = -self.args['server_lr']
        sm_of_signs[sm_of_signs >= self.robustLR_threshold] = self.args['server_lr']                                            
        return sm_of_signs
        
            
    def agg_avg(self, agent_updates_dict, num_dps):
        """ classic fed avg """
        sm_updates, total_data = 0, 0
        for _id, update in enumerate(agent_updates_dict):
            n_agent_data = num_dps[_id]
            sm_updates +=  n_agent_data * update
            total_data += n_agent_data  
        return  sm_updates / total_data
    
    def agg_comed(self, agent_updates_dict):
        agent_updates_col_vector = [update.view(-1, 1) for update in agent_updates_dict.values()]
        concat_col_vectors = torch.cat(agent_updates_col_vector, dim=1)
        return torch.median(concat_col_vectors, dim=1).values
    
    def agg_sign(self, agent_updates_dict):
        """ aggregated majority sign update """
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]
        sm_signs = torch.sign(sum(agent_updates_sign))
        return torch.sign(sm_signs)


class FLAME(Defense):
    def __init__(self, num_workers):
        self.join_clients = num_workers

    def exec(self, global_model, client_models):
        # 0. 预处理
        vectorize_models = [vectorize_net(model).clone().detach() for model in client_models]
        models_matrix = torch.stack(vectorize_models)
        vectorize_global_model = vectorize_net(global_model).clone().detach()
        vectorize_updates = models_matrix - vectorize_global_model
        l2_norm = torch.linalg.norm(vectorize_updates, dim = 1)
        median = torch.median(l2_norm)
        
        # 1. 聚类
        models_matrix = models_matrix.double().cpu()
        cluster = hdbscan.HDBSCAN(metric="cosine", algorithm="generic", min_cluster_size=self.join_clients//2+1, min_samples=1,allow_single_cluster=True)
        cluster.fit(models_matrix)

        # 2. 范数中值裁剪
        gama = median.div(l2_norm)
        scale = torch.min(torch.tensor(1), gama)
        vectorize_updates.mul_(scale.unsqueeze(1))

        # 3. 聚合
        uploaded_models = []
        uploaded_weights = []
        for i, data in enumerate(vectorize_updates):
            if cluster.labels_[i] == 0:
                print (i, end=' ')
                uploaded_model = data + vectorize_global_model
                tm_model = copy.deepcopy(global_model)
                load_model_weight(tm_model, uploaded_model)
                uploaded_models.append(tm_model)
        print ()
        
        l = len(uploaded_models)
        client_models = uploaded_models
        uploaded_weights = [1 / l] * l
        return client_models, uploaded_weights, median