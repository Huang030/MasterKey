import sys
sys.path.append("../")
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import random
import numpy as np
import copy
import os
from .attacker import Attacker, A3FL_MT, CerP_MT, Neu_MT, PGD_MT
from .aggregator import Aggregator
from .invert_CIFAR import trigger_fast_train
from .utils import apply_grad_mask, model_dist_norm_var, parameters_to_vector, vector_to_parameters, init_masks, calculate_sparsities, Agent_s
from math import ceil
import pickle
import time

class FLer:
    def __init__(self, helper):
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

        self.helper = helper
        
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)
        self.aggregator = Aggregator(self.helper)
        self.attacker_criterion = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)
        if self.helper.config.is_poison:
            attacks = {"MasterKey": Attacker, "A3FL_MT": A3FL_MT, "CerP_MT": CerP_MT, "Neu_MT": Neu_MT, "PGD_MT": PGD_MT}
            self.attacker = attacks[self.helper.config.attacker_method](self.helper)
        else:
            self.attacker = None
        self.save_attack_model = True
        self.setup_save_path()

        if self.helper.config.load_benign_model:
            model_path = f'../saved/benign/{self.helper.config.dataset}_{self.helper.config.poison_start_epoch}_{self.helper.config.agg_method}_{self.helper.config.num_total_participants}.pt'
            # model_path = f'../saved/benign/{self.helper.config.dataset}_{self.helper.config.poison_start_epoch}_{self.helper.config.agg_method}_{self.helper.config.num_total_participants}_{self.helper.config.car}.pt'
            # model_path = '../saved/benign/tiny-imagenet_1799_avg_100.pt'
            self.helper.global_model.load_state_dict(torch.load(model_path, map_location = 'cuda')['model'])
            loss,acc = self.test_once()
            print(f'Load benign model {model_path}, acc {acc:.3f}')

        if self.helper.config.agg_method == "lockdown":
            params = {name: copy.deepcopy(self.helper.global_model.state_dict()[name]) for name in self.helper.global_model.state_dict()}
            self.sparsity = calculate_sparsities(0.25, params, distribution="ERK")
            mask = init_masks(params, self.sparsity)
            self.agents = []
            for _id in range(0, self.helper.config.num_total_participants):
                agent = Agent_s(_id, self.helper, mask=mask)
                self.agents.append(agent)
            self.global_mask = {}
            self.updates_dict = {}

    def setup_save_path(self):
        self.images_save_path = f'../saved/images/eps_{self.helper.config.original_eps}_atkepochs_{self.helper.config.atk_model_epochs}/'
        if not os.path.exists(self.images_save_path):
            os.makedirs(self.images_save_path)

    def log_once(self, epoch, loss, acc, bkd_loss, bkd_acc):
        log_dict = {
            'epoch': epoch, 
            'test_acc': acc,
            'test_loss': loss, 
        }
        if isinstance(bkd_loss, list): 
            log_dict['bkd_loss'] = sum(bkd_loss) / len(bkd_loss)
            log_dict['bkd_acc'] = sum(bkd_acc) / len(bkd_acc)
        else:  
            log_dict['bkd_loss'] = bkd_loss
            log_dict['bkd_acc'] = bkd_acc
        wandb.log(log_dict)
        print("=====>Global Model Test<=====")
        print('|'.join([f'{k}:{float(log_dict[k]):.3f}' for k in log_dict if isinstance(log_dict[k], (int, float))]))
        if isinstance(bkd_loss, list):
            for i, (loss, acc) in enumerate(zip(bkd_loss, bkd_acc)):
                print(f"Target {i} - bkd_loss: {loss:.3f}, bkd_acc: {acc:.3f}")
        print()
        self.save_model(epoch, log_dict)

    def save_model(self, epoch, log_dict):
        if self.helper.config.is_poison and epoch == self.helper.config.poison_epochs - 1:
            if self.helper.config.attacker_method == 'MasterKey':
                torch.save(self.helper.global_model.state_dict(),
                            f'../saved/poison/{self.helper.config.dataset}_{self.helper.config.attacker_method}_poisoned.pt')
                torch.save(self.attacker.atk_model.state_dict(), 
                            f"../saved/poison/{self.helper.config.dataset}_{self.helper.config.original_eps}_atkmodel")
            if self.helper.config.attacker_method == 'A3FL_MT':
                torch.save(self.helper.global_model.state_dict(),
                            f'../saved/poison/{self.helper.config.dataset}_{self.helper.config.attacker_method}_poisoned.pt')
                torch.save(self.attacker.triggers,
                            f'../saved/poison/{self.helper.config.dataset}_{self.helper.config.attacker_method}_trigger.pt')
            if self.helper.config.attacker_method == 'Neu_MT':
                torch.save(self.helper.global_model.state_dict(),
                            f'../saved/poison/{self.helper.config.dataset}_{self.helper.config.attacker_method}_poisoned.pt')
                torch.save(self.attacker.triggers,
                            f'../saved/poison/{self.helper.config.dataset}_{self.helper.config.attacker_method}_trigger.pt')
            print(f'Attack Model saved')
        elif epoch == self.helper.config.poison_epochs - 1:
            torch.save(self.helper.global_model.state_dict(),
                        f'../saved/poison/{self.helper.config.dataset}_benign.pt')


        if epoch > 1700 and (epoch + 1) % self.helper.config.save_every == 0:
            log_dict['model'] = self.helper.global_model.state_dict()
            if self.helper.config.is_poison:
                pass
            else:
                assert self.helper.config.lr_method == 'fix-lr'
                save_path = f'../saved/benign/{self.helper.config.dataset}_{epoch}_{self.helper.config.agg_method}_{self.helper.config.num_total_participants}.pt'
                torch.save(log_dict, save_path)
                print(f'Model saved at {save_path}')

    def test_once(self, poison = False, epoch = None, target = None):
        model = self.helper.global_model
        if self.helper.config.agg_method == "lockdown":
            model = copy.deepcopy(self.helper.global_model)
            for name, param in model.named_parameters():
                mask = 0
                for id, agent in enumerate(self.agents):
                    mask += self.old_masks[id][name].cuda()
                param.data = torch.where(mask.cuda() >= 20, param,
                                            torch.zeros_like(param))
                # print(torch.sum(mask.cuda() >= 20) / torch.numel(mask))
        model.eval()
        with torch.no_grad():               
            data_source = self.helper.test_data
            total_loss = 0
            correct = 0
            num_data = 0.
            for batch_id, batch in enumerate(data_source):
                data, targets = batch
                data, targets = data.cuda(), targets.cuda()
                clean_img = data.clone()
                if poison:
                    if target == None:
                        data, targets = self.attacker.poison_input(data, targets, eval=True)
                        atkdata = data.clone()
                    else:
                        data, targets = self.attacker.poison_input(data, targets, eval=True, target=target)
                        atkdata = data.clone()
                output = model(data)
                total_loss += self.criterion(output, targets).item()
                pred = output.data.max(1)[1] 
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
                num_data += output.size(0) 
        acc = float(correct) / float(num_data)
        loss = total_loss
        model.train()

        if poison and self.helper.config.save_imgs:
            clean_img, poison_img = clean_img[:10].clone().cpu(), atkdata[:10].clone().cpu()
            residual = poison_img-clean_img
            clean_img = F.upsample(clean_img, scale_factor=(4, 4))
            poison_img = F.upsample(poison_img, scale_factor=(4, 4))
            residual = F.upsample(residual, scale_factor=(4, 4))
            
            
            all_img = torch.cat([clean_img, residual, poison_img], 0)
            grid = torchvision.utils.make_grid(all_img.clone(), nrow=10, normalize=True)

            torchvision.utils.save_image(
                grid, os.path.join(self.images_save_path, 
                                    f'{epoch}_all_images.png'))
            torchvision.utils.save_image(
                torchvision.utils.make_grid(
                    clean_img.clone(), nrow=10, normalize=True), 
                os.path.join(self.images_save_path, 
                                f'{epoch}_clean_images.png'))
            torchvision.utils.save_image(
                torchvision.utils.make_grid(
                    residual.clone(), nrow=10), 
                os.path.join(self.images_save_path,  f'{epoch}_residual.png'))
            torchvision.utils.save_image(
                torchvision.utils.make_grid(
                    poison_img.clone(), nrow=10, normalize=True), 
                os.path.join(self.images_save_path, 
                                f'{epoch}_poison_images.png'))

        return loss, acc

    def train(self):
        for epoch in range(-2, self.helper.config.epochs):
            print (f'{self.helper.config.attacker_method} Training Epoch: {epoch} / {self.helper.config.epochs - 1}')
            sampled_participants = self.sample_participants(epoch)
            print (f"Sampled_participants: {sampled_participants}")
            local_models, local_weights = self.train_once(epoch, sampled_participants)
            self.aggregator.agg(self.helper.global_model, local_models, local_weights)
            loss, acc = self.test_once()
            if self.helper.config.eval_mode == 'random':
                bkd_loss, bkd_acc = self.test_once(poison = self.helper.config.is_poison, epoch=epoch)
                self.log_once(epoch, loss, acc, bkd_loss, bkd_acc)
            elif self.helper.config.eval_mode == 'target':
                bkd_losses, bkd_accs = [], []
                for target in range(self.helper.config.num_classes):
                    bkd_loss, bkd_acc = self.test_once(poison = self.helper.config.is_poison, epoch=epoch, target=target)
                    bkd_losses.append(bkd_loss)
                    bkd_accs.append(bkd_acc)
                self.log_once(epoch, loss, acc, bkd_losses, bkd_accs)

    def train_once(self, epoch, sampled_participants):
        if self.helper.config.agg_method == "lockdown":
            self.old_masks = [copy.deepcopy(agent.mask) for agent in self.agents]
        local_models = []
        local_weights = []
        mask_grad_list = None
        local_model = self.helper.local_model
        global_model = self.helper.global_model

        first_adversary, adv = self.contain_adversary(epoch, sampled_participants)
        print (f"fisrt adversary: {first_adversary}")
        if first_adversary >= 0:
            model = local_model
            self.copy_params(model, global_model)
            indices = []
            for ad in adv:
                indice = list(self.helper.train_data[ad].sampler)
                indices.extend(indice)
            merged_dataloader = torch.utils.data.DataLoader(
                self.helper.train_dataset,
                batch_size=self.helper.config.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
                num_workers=self.helper.config.num_worker)
            
            if self.helper.config.attacker_method == 'PGD_MT':
                pass

            elif self.helper.config.attacker_method == 'Neu_MT':
                mask_grad_list = self.attacker.grad_mask(copy.deepcopy(model), self.helper.train_data[first_adversary])
            
            elif self.helper.config.attacker_method == 'CerP_MT':
                self.attacker.search_trigger(model, self.helper.train_data[first_adversary], epoch)
            
            elif self.helper.config.attacker_method == 'A3FL_MT':
                self.attacker.search_trigger(model, self.helper.train_data[first_adversary], epoch)
            
            elif self.helper.config.attacker_method == 'MasterKey':
                self.attacker.search_trigger(model, merged_dataloader, epoch)

            else:
                raise NotImplementedError
            
        malicious_model = None
        act_par = []
        for participant_id in sampled_participants:
            model = local_model
            self.copy_params(model, global_model)
            model.train()
            if not self.helper.config.is_poison or not self.if_adversary(epoch, participant_id, sampled_participants):
                if self.helper.config.agg_method == "flip":
                    pass
                elif self.helper.config.agg_method == "lockdown":
                    self.agents[participant_id].local_train(model, participant_id, epoch, global_mask=self.global_mask, 
                                                            neurotoxin_mask = None, updates_dict=self.updates_dict)
                else:
                    self.train_benign(participant_id, model, epoch)
            elif malicious_model == None:
                print (f"Malicious attack: {participant_id}")
                self.train_malicious(participant_id, model, epoch, mask_grad_list)
                malicious_model = copy.deepcopy(model)
            else:
                model = malicious_model
                print (f"Supporting attack: {participant_id}")
            
            # r = random.random()
            # if participant_id == 0:
            #     print (r, self.helper.config.car)
            # if r < self.helper.config.car:       
            #     act_par.append(participant_id)
            local_models.append(copy.deepcopy(model))
            local_weights.append(1.0)
        # print (f"Active participants this round: {act_par}")
        local_weights = [i/sum(local_weights) for i in local_weights]
        return local_models, local_weights

    def train_malicious(self, participant_id, model, epoch, mask_grad_list=None):
        if self.helper.config.attacker_method == 'PGD_MT':
            model_original = list(copy.deepcopy(model).parameters())

        if self.helper.config.attacker_method == 'CerP_MT':
            normalmodel = copy.deepcopy(model)
            self.train_benign(participant_id, normalmodel, epoch)
            normal_params_variables = dict()
            for name, param in normalmodel.named_parameters():
                normal_params_variables[name] = normalmodel.state_dict()[name].clone().detach().requires_grad_(
                    False)

        lr = self.get_lr(epoch)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
            momentum=self.helper.config.momentum,
            weight_decay=self.helper.config.decay)
        for internal_epoch in range(self.helper.config.attacker_retrain_times):
            for batch_idx, (inputs, labels) in enumerate(self.helper.train_data[participant_id]):
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = self.attacker.poison_input(inputs, labels)
                output = model(inputs)
                loss = self.attacker_criterion(output, labels)

                if self.helper.config.attacker_method == 'CerP_MT':
                    loss = loss + 0.0001 * model_dist_norm_var(model, normal_params_variables)

                optimizer.zero_grad()
                loss.backward()

                if self.helper.config.attacker_method == 'Neu_MT' and mask_grad_list:
                    apply_grad_mask(model, mask_grad_list)

                optimizer.step()

                if self.helper.config.attacker_method == 'PGD_MT':
                    w = list(model.parameters())
                    w_vec = parameters_to_vector(w)
                    model_original_vec = parameters_to_vector(model_original)
                    print (torch.norm(w_vec - model_original_vec))
                    if (batch_idx%self.attacker.project_frequency == 0 or batch_idx == len(self.helper.train_data[participant_id])-1) and (torch.norm(w_vec - model_original_vec) > self.attacker.eps):
                        w_proj_vec = self.attacker.eps*(w_vec - model_original_vec)/torch.norm(
                                w_vec-model_original_vec) + model_original_vec
                        vector_to_parameters(w_proj_vec, w)
                        print (f"PGD, {torch.norm(w_proj_vec - model_original_vec)}")

    def train_benign(self, participant_id, model, epoch):
        lr = self.get_lr(epoch)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
            momentum=self.helper.config.momentum,
            weight_decay=self.helper.config.decay)
        for internal_epoch in range(self.helper.config.retrain_times):
            for inputs, labels in self.helper.train_data[participant_id]:
                inputs, labels = inputs.cuda(), labels.cuda()
                output = model(inputs)
                loss = self.criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 
        
    def contain_adversary(self, epoch, sampled_participants):
        if self.helper.config.is_poison and \
            epoch < self.helper.config.poison_epochs and epoch >= 0:
            adv = []
            if self.helper.config.sample_method == 'fix':
                for p in sampled_participants:
                    if p < self.helper.config.num_adversaries:
                        adv.append(p)
                return adv[0], adv
            else:
                raise NotImplementedError
        return -1, []

    def if_adversary(self, epoch, participant_id, sampled_participants):
        if self.helper.config.is_poison and epoch < self.helper.config.poison_epochs and epoch >= 0:
            if self.helper.config.sample_method == 'fix':
                if participant_id < self.helper.config.num_adversaries:
                    return True 
                else:
                    return False
            else:
                raise NotImplementedError
        else:
            return False

    def create_global_model_copy(self):
        global_model_copy = dict()
        for name, param in self.helper.global_model.named_parameters():
            global_model_copy[name] = self.helper.global_model.state_dict()[name].clone().detach()
        return global_model_copy

    def create_weight_accumulator(self):
        weight_accumulator = dict()
        for name, data in self.helper.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(data)
        return weight_accumulator
    
    def update_weight_accumulator(self, model, weight_accumulator):
        for name, data in model.state_dict().items():
            weight_accumulator[name].add_(data - self.helper.global_model.state_dict()[name])
        return weight_accumulator

    def get_lr(self, epoch):
        if self.helper.config.lr_method == 'fix-lr':
            lr = self.helper.config.lr

        elif self.helper.config.lr_method == 'exp':
            tmp_epoch = epoch
            if self.helper.config.is_poison and self.helper.config.load_benign_model:
                tmp_epoch += self.helper.config.poison_start_epoch
            lr = self.helper.config.lr * (self.helper.config.gamma**tmp_epoch)

        elif self.helper.config.lr_method == 'linear':
            if self.helper.config.is_poison or epoch > 1900:
                lr = 0.002
            else:
                lr_init = self.helper.config.lr
                target_lr = self.helper.config.target_lr
                if epoch <= self.helper.config.epochs/2.:
                    lr = epoch*(target_lr - lr_init)/(self.helper.config.epochs/2.-1) + lr_init - (target_lr - lr_init)/(self.helper.config.epochs/2. - 1)
                else:
                    lr = (epoch-self.helper.config.epochs/2)*(-target_lr)/(self.helper.config.epochs/2) + target_lr

                if lr <= 0.002:
                    lr = 0.002
                # else:
                #     raise NotImplementedError
        return lr

    def sample_participants(self, epoch):
        if self.helper.config.sample_method == 'random':
            sampled_participants = random.sample(
                range(self.helper.config.num_total_participants), 
                self.helper.config.num_sampled_participants)
        elif self.helper.config.sample_method == 'fix':
            if (self.helper.config.is_poison and epoch < self.helper.config.poison_epochs and epoch >= 0):
                sampled_participants = list(range(self.helper.config.num_adversaries))
                benign_participants = list(random.sample(range(self.helper.config.num_adversaries, self.helper.config.num_total_participants),
                                                            self.helper.config.num_sampled_participants - self.helper.config.num_adversaries))
                sampled_participants.extend(benign_participants)
            else:
                sampled_participants = random.sample(
                    range(self.helper.config.num_total_participants), 
                    self.helper.config.num_sampled_participants)
        else:
            raise NotImplementedError
        assert len(sampled_participants) == self.helper.config.num_sampled_participants
        return sampled_participants
    
    def copy_params(self, model, new_model):
        for old_param, new_param in zip(model.parameters(), new_model.parameters()):
            old_param.data = new_param.data.clone()
        
        