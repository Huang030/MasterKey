import sys
sys.path.append("../")
import wandb
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision import datasets
from torchvision import datasets, transforms
from collections import defaultdict, OrderedDict
import random
import numpy as np
from models.resnet import ResNet18
import copy
import os
from sklearn.cluster import DBSCAN
from .defense import AddNoise, WeightDiffClippingDefense, Krum, RLR, FLAME

def fed_avg_aggregator(init_model, net_list, net_freq):
    weight_accumulator = {}
    
    for name, params in init_model.state_dict().items():
        weight_accumulator[name] = torch.zeros_like(params).float()
    
    for i in range(0, len(net_list)):
        diff = dict()
        for name, data in net_list[i].state_dict().items():
            diff[name] = (data - init_model.state_dict()[name])
            try:
                weight_accumulator[name].add_(net_freq[i]  *  diff[name])
                
            except Exception as e:
                print(e)
                import IPython
                IPython.embed()
                exit(0)
    for idl, (name, data) in enumerate(init_model.state_dict().items()):
        update_per_layer = weight_accumulator[name]
        if data.type() != update_per_layer.type():
            data.add_(update_per_layer.to(torch.int64))            
        else:
            data.add_(update_per_layer)
            
    return init_model


class Aggregator:
    def __init__(self, helper):
        self.helper = helper
        self.setup_defense()

    def setup_defense(self):
        if self.helper.config.agg_method == "avg":
            self._defender = None

        elif self.helper.config.agg_method == "norm-clipping" or self.helper.config.agg_method == "norm-clipping-adaptive":
            # self._defender = WeightDiffClippingDefense(norm_bound=1)
            self._defender = WeightDiffClippingDefense(norm_bound=0.5)

        elif self.helper.config.agg_method == "weak-dp":
            # doesn't really add noise. just clips
            # self._defender = WeightDiffClippingDefense(norm_bound=2)
            self._defender = WeightDiffClippingDefense(norm_bound=0.8)

        elif self.helper.config.agg_method == "krum":
            self._defender = Krum(mode='krum', num_workers=self.helper.config.num_sampled_participants, num_adv=self.helper.config.num_adversaries)

        elif self.helper.config.agg_method == "multi-krum":
            self._defender = Krum(mode='multi-krum', num_workers=self.helper.config.num_sampled_participants, num_adv=self.helper.config.num_adversaries)
        
        elif self.helper.config.agg_method == "rlr":
            pytorch_total_params = sum(p.numel() for p in self.helper.global_model.parameters())
            args_rlr={
                'aggr':'avg',
                'noise':0,
                'clip': 0,
                'server_lr': 1.0,
            }
            theta = 5
            self._defender = RLR(n_params=pytorch_total_params, args=args_rlr, robustLR_threshold=theta)
        
        elif self.helper.config.agg_method == "flmae":
            self._defender = FLAME(num_workers=self.helper.config.num_sampled_participants)

        elif self.helper.config.agg_method == "rfa":
            NotImplementedError("Unsupported defense method !")

        elif self.helper.config.agg_method == "crfl":
            NotImplementedError("Unsupported defense method !")


        elif self.helper.config.agg_method == "foolsgold":
            NotImplementedError("Unsupported defense method !")
            # pytorch_total_params = sum(p.numel() for p in self.net_avg.parameters())
            # self._defender = FoolsGold(num_clients=self.part_nets_per_round, num_classes=10, num_features=pytorch_total_params)

        else:
            NotImplementedError("Unsupported defense method !")

    def agg(self, net_avg, net_list, net_freq):
        if self.helper.config.agg_method == 'avg':
            pass

        elif self.helper.config.agg_method == "norm-clipping":
            for net_idx, net in enumerate(net_list):
                self._defender.exec(client_model=net, global_model=net_avg)

        elif self.helper.config.agg_method == "norm-clipping-adaptive":
            # we will need to adapt the norm diff first before the norm diff clipping
            # logger.info("#### Let's Look at the Nom Diff Collector : {} ....; Mean: {}".format(norm_diff_collector, 
            #     np.mean(norm_diff_collector)))
            # self._defender.norm_bound = np.mean(norm_diff_collector)
            # for net_idx, net in enumerate(net_list):
            #     self._defender.exec(client_model=net, global_model=self.net_avg)
            raise NotImplementedError

        elif self.helper.config.agg_method == "weak-dp":
            # this guy is just going to clip norm. No noise added here
            for net_idx, net in enumerate(net_list):
                self._defender.exec(client_model=net, global_model=net_avg)

        elif self.helper.config.agg_method == 'krum':
            net_list, net_freq = self._defender.exec(client_models=net_list, 
                                                    num_dps=net_freq)
            
        elif self.helper.config.agg_method == 'multi-krum':
            net_list, net_freq = self._defender.exec(client_models=net_list, 
                                                    num_dps=net_freq)

        elif self.helper.config.agg_method == 'rlr':
            net_list, net_freq = self._defender.exec(client_models=net_list,
                                                    global_model=copy.deepcopy(net_avg),
                                                    num_dps=net_freq)

        elif self.helper.config.agg_method == 'flmae':
            net_list, net_freq, median = self._defender.exec(global_model=net_avg, client_models=net_list)
        
        elif self.helper.config.agg_method == 'Median':
            pass
        elif self.helper.config.agg_method == 'Trimmed-Mean':
            pass
        
        fed_avg_aggregator(net_avg, net_list, net_freq)

        if self.helper.config.agg_method == "weak-dp":
            # add noise to net_avg
            # noise_adder = AddNoise(stddev=0.002)
            noise_adder = AddNoise(stddev=0.001)
            noise_adder.exec(client_model=net_avg)
        elif self.helper.config.agg_method == "flmae":
            lamda = 0.001
            std = median * lamda
            # add noise to net_avg
            noise_adder = AddNoise(stddev=std)
            noise_adder.exec(client_model=net_avg)