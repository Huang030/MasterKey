import torch
import numpy as np
import hdbscan
import copy
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from .utils import LocalUpdate
from sklearn.cluster import DBSCAN


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


class RFA(Defense):
    """
    we implement the robust aggregator at: 
    https://arxiv.org/pdf/1912.13445.pdf
    the code is translated from the TensorFlow implementation: 
    https://github.com/krishnap25/RFA/blob/01ec26e65f13f46caf1391082aa76efcdb69a7a8/models/model.py#L264-L298
    """

    def __init__(self, *args, **kwargs):
        pass

    def exec(self, client_models, net_freq,
             maxiter=4, eps=1e-5,
             ftol=1e-6, device=torch.device("cuda"),
             *args, **kwargs):
        """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
        """
        alphas = torch.tensor(net_freq, dtype=torch.float32, device=device)
        vectorize_nets = [vectorize_net(cm).detach() for cm in client_models]
        median = self.weighted_average_oracle(vectorize_nets, alphas)

        num_oracle_calls = 1

        # logging
        obj_val = self.geometric_median_objective(median=median, points=vectorize_nets, alphas=alphas)

        logs = []
        log_entry = [0, obj_val, 0, 0]
        logs.append("Tracking log entry: {}".format(log_entry))
        print ('Starting Weiszfeld algorithm')
        print (log_entry)

        # start
        for i in range(maxiter):
            prev_median, prev_obj_val = median, obj_val
            weights = torch.tensor([alpha / max(eps, self.l2dist(median, p)) for alpha, p in zip(alphas, vectorize_nets)],
                                   dtype=alphas.dtype, device=device)
            weights = weights / weights.sum()
            median = self.weighted_average_oracle(vectorize_nets, weights)
            num_oracle_calls += 1
            obj_val = self.geometric_median_objective(median, vectorize_nets, alphas)
            log_entry = [i+1, obj_val,
                         (prev_obj_val - obj_val)/obj_val,
                         self.l2dist(median, prev_median)]
            logs.append(log_entry)
            logs.append("Tracking log entry: {}".format(log_entry))
            print ("#### Oracle Cals: {}, Objective Val: {}".format(num_oracle_calls, obj_val))
            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                break
        #logger.info("Num Oracale Calls: {}, Logs: {}".format(num_oracle_calls, logs))

        aggregated_model = client_models[0]  # create a clone of the model
        load_model_weight(aggregated_model, median.to(device))
        neo_net_list = [aggregated_model]
        neo_net_freq = [1.0]
        return neo_net_list, neo_net_freq

    def weighted_average_oracle(self, points, weights):
        """Computes weighted average of atoms with specified weights
        Args:
            points: list, whose weighted average we wish to calculate
                Each element is a list_of_torch.Tensor
            weights: list of weights of the same length as atoms
        """
        tot_weights = weights.sum()
        weighted_updates = torch.zeros(points[0].shape, dtype=points[0].dtype, device=points[0].device)
        for w, p in zip(weights, points):
            weighted_updates += (w * p / tot_weights)
        return weighted_updates

    def l2dist(self, p1, p2):
        """L2 distance between p1, p2, each of which is a list of nd-arrays"""
        return torch.norm(p1 - p2)

    def geometric_median_objective(self, median, points, alphas):
        """Compute geometric median objective."""
        return torch.sum(torch.stack([alpha * self.l2dist(median, p) for alpha, p in zip(alphas, points)]))


class Median:
    def __init__(self):
        pass

    def exec(self, client_models, net_freq):
        vectorize_models = [vectorize_net(model).clone().detach() for model in client_models]
        stacked_vector = torch.stack(vectorize_models)
        median_vector = torch.median(stacked_vector, dim=0)[0]
        aggregated_model = client_models[0]
        load_model_weight(aggregated_model, median_vector)
        return [aggregated_model], [1.0]

def kernel_function(x, y, sigma=1.0):
    # 向量化计算核函数：x.shape=(m,d), y.shape=(n,d) -> 输出 (m,n)
    pairwise_dist = torch.cdist(x, y, p=2)  # 计算欧氏距离矩阵 (m,n)
    return torch.exp(-pairwise_dist ** 2 / (2 * sigma ** 2))

def compute_mmd(x, y, sigma=1.0):
    m, n = x.size(0), y.size(0)
    # 计算核矩阵 (避免显式循环)
    K_xx = kernel_function(x, x, sigma)  # (m,m)
    K_yy = kernel_function(y, y, sigma)  # (n,n)
    K_xy = kernel_function(x, y, sigma)  # (m,n)
    
    # 计算 MMD (排除对角线元素)
    mmd = (K_xx.sum() - K_xx.diag().sum()) / (m * (m - 1)) + \
          (K_yy.sum() - K_yy.diag().sum()) / (n * (n - 1)) - \
          2 * K_xy.mean()
    return mmd


def flare(w_updates, w_locals, net, central_dataset, dataset_test, global_parameters, helper):
    w_feature=[]
    temp_model = copy.deepcopy(net)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    for client in w_locals:
        net.load_state_dict(client)
        local = LocalUpdate(
                helper=helper, dataset=dataset_test, idxs=central_dataset)
        feature = local.get_PLR(
            net=copy.deepcopy(net).cuda())
        w_feature.append(feature)
    distance_list=[[] for i in range(len(w_updates))]
    # distance_list=[list(len(w_updates)) for i in range(len(w_updates))]
    for i in range(len(w_updates)):
        for j in range(i+1, len(w_updates)):
            score = compute_mmd(w_feature[i], w_feature[j])
            distance_list[i].append(score.item())
            distance_list[j].append(score.item())
    print('defense line121 distance_list', distance_list)
    vote_counter=[0 for i in range(len(w_updates))]
    k = round(len(w_updates)*0.5)
    for i in range(len(w_updates)):
        IDs = np.argsort(distance_list[i])
        for j in range(len(IDs)):
            # client_id is the index of client i-th client voting for
            # distance_list[] only records score with other clients without itself
            # so distance_list[i][i] should be itself
            # client_id = j + 1 after j >= i
            if IDs[j] >= i:
                client_id = IDs[j] + 1 
            else:
                client_id = IDs[j]
            vote_counter[client_id] += 1
            if j + 1 >= k:  # first 𝑘 elements in 𝐼 𝐷𝑠 and vote for it
                break

    trust_score = [x/sum(vote_counter) for x in vote_counter]
    # print('defense line188 len trust_score', trust_score)
    return trust_score


def deepsight_aggregate_global_model(net_glob, clients, net_freq, chosen_ids):
    def ensemble_cluster(neups, ddifs, biases):
        biases = np.array([bias.cpu().numpy() for bias in biases])
        #neups = np.array([neup.cpu().numpy() for neup in neups])
        #ddifs = np.array([ddif.cpu().detach().numpy() for ddif in ddifs])
        N = len(neups)
        # use bias to conduct DBSCAM
        # biases= np.array(biases)
        cosine_labels = DBSCAN(min_samples=3,metric='cosine').fit(biases).labels_
        print("cosine_cluster:{}".format(cosine_labels))
        # neups=np.array(neups)
        neup_labels = DBSCAN(min_samples=3).fit(neups).labels_
        print("neup_cluster:{}".format(neup_labels))
        ddif_labels = DBSCAN(min_samples=3).fit(ddifs).labels_
        print("ddif_cluster:{}".format(ddif_labels))

        dists_from_cluster = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                dists_from_cluster[i, j] = (int(cosine_labels[i] == cosine_labels[j]) + int(
                    neup_labels[i] == neup_labels[j]) + int(ddif_labels[i] == ddif_labels[j]))/3.0
                dists_from_cluster[j, i] = dists_from_cluster[i, j]
                
        print("dists_from_clusters:")
        print(dists_from_cluster)
        ensembled_labels = DBSCAN(min_samples=3,metric='precomputed').fit(dists_from_cluster).labels_

        return ensembled_labels
    
    global_weight = list(net_glob.state_dict().values())[-2]
    global_bias = list(net_glob.state_dict().values())[-1]

    biases = [(list(clients[i].state_dict().values())[-1] - global_bias) for i in chosen_ids]
    weights = [list(clients[i].state_dict().values())[-2] for i in chosen_ids]

    n_client = len(chosen_ids)
    cosine_similarity_dists = np.array((n_client, n_client))
    neups = list()
    n_exceeds = list()

    # calculate neups
    sC_nn2 = 0
    for i in range(len(chosen_ids)):
        C_nn = torch.sum(weights[i]-global_weight, dim=[1]) + biases[i]-global_bias
        # print("C_nn:",C_nn)
        C_nn2 = C_nn * C_nn
        neups.append(C_nn2)
        sC_nn2 += C_nn2
        
        C_max = torch.max(C_nn2).item()
        threshold = 0.01 * C_max if 0.01 > (1 / len(biases)) else 1 / len(biases) * C_max
        n_exceed = torch.sum(C_nn2 > threshold).item()
        n_exceeds.append(n_exceed)
    # normalize
    neups = np.array([(neup/sC_nn2).cpu().numpy() for neup in neups])
    print("n_exceeds:{}".format(n_exceeds))
    # 256 can be replaced with smaller value
    rand_input = torch.randn((256, 3, 32, 32)).cuda()

    global_ddif = torch.mean(torch.softmax(net_glob(rand_input), dim=1), dim=0)
    # print("global_ddif:{} {}".format(global_ddif.size(),global_ddif))
    client_ddifs = [torch.mean(torch.softmax(clients[i](rand_input), dim=1), dim=0)/ global_ddif
                    for i in chosen_ids]
    client_ddifs = np.array([client_ddif.cpu().detach().numpy() for client_ddif in client_ddifs])
    # print("client_ddifs:{}".format(client_ddifs[0]))

    # use n_exceed to label
    classification_boundary = np.median(np.array(n_exceeds)) / 2
    
    identified_mals = [int(n_exceed <= classification_boundary) for n_exceed in n_exceeds]
    print("identified_mals:{}".format(identified_mals))
    clusters = ensemble_cluster(neups, client_ddifs, biases)
    print("ensemble clusters:{}".format(clusters))
    cluster_ids = np.unique(clusters)

    deleted_cluster_ids = list()
    for cluster_id in cluster_ids:
        n_mal = 0
        cluster_size = np.sum(cluster_id == clusters)
        for identified_mal, cluster in zip(identified_mals, clusters):
            if cluster == cluster_id and identified_mal:
                n_mal += 1
        print("cluser size:{} n_mal:{}".format(cluster_size,n_mal))        
        if (n_mal / cluster_size) >= (1 / 3):
            deleted_cluster_ids.append(cluster_id)
    # print("deleted_clusters:",deleted_cluster_ids)
    temp_chosen_ids = copy.deepcopy(chosen_ids)
    for i in range(len(chosen_ids)-1, -1, -1):
        # print("cluster tag:",clusters[i])
        if clusters[i] in deleted_cluster_ids:
            del chosen_ids[i]

    print("final clients length:{}".format(len(chosen_ids)))
    if len(chosen_ids)==0:
        chosen_ids = temp_chosen_ids
    
    net_freq = [0] * len(net_freq)
    for i in chosen_ids:
        net_freq[i] = 1/len(chosen_ids)

    return clients, net_freq

