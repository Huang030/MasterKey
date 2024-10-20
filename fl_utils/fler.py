import sys
sys.path.append("../")
import wandb
import torch
import random
import numpy as np
import copy
import os
from .attacker import Attacker
from .aggregator import Aggregator
from math import ceil
import pickle

class FLer:
    def __init__(self, helper):
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

        self.helper = helper
        
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)
        self.aggregator = Aggregator(self.helper)
        self.attacker_criterion = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)
        if self.helper.config.is_poison:
            self.attacker = Attacker(self.helper)
        else:
            self.attacker = None
        if self.helper.config.sample_method == 'random_updates':
            self.init_advs()
        if self.helper.config.load_benign_model:
            model_path = f'../saved/benign_new/{self.helper.config.dataset}_{self.helper.config.poison_start_epoch}_{self.helper.config.lr_method}.pt'
            self.helper.global_model.load_state_dict(torch.load(model_path, map_location = 'cuda')['model'])
            loss,acc = self.test_once()
            print(f'Load benign model {model_path}, acc {acc:.3f}')
        return
    
    def init_advs(self):
        num_updates = self.helper.config.num_sampled_participants * self.helper.config.poison_epochs
        num_poison_updates = ceil(self.helper.config.sample_poison_ratio * num_updates)
        updates = list(range(num_updates))
        advs = np.random.choice(updates, num_poison_updates, replace=False)
        print(f'Using random updates, sampled {",".join([str(x) for x in advs])}')
        adv_dict = {}
        for adv in advs:
            epoch = adv//self.helper.config.num_sampled_participants
            idx = adv % self.helper.config.num_sampled_participants
            if epoch in adv_dict:
                adv_dict[epoch].append(idx)
            else:
                adv_dict[epoch] = [idx]
        self.advs = adv_dict

    def log_once(self, epoch, loss, acc, bkd_loss, bkd_acc):
        log_dict = {
            'epoch': epoch, 
            'test_acc': acc,
            'test_loss': loss, 
            'bkd_acc': bkd_acc,
            'bkd_loss': bkd_loss
            }
        wandb.log(log_dict)
        print('|'.join([f'{k}:{float(log_dict[k]):.3f}' for k in log_dict]))
        print ()
        self.save_model(epoch, log_dict)

    def save_model(self, epoch, log_dict):
        if (epoch + 1) % self.helper.config.save_every == 0:
            log_dict['model'] = self.helper.global_model.state_dict()
            if self.helper.config.is_poison:
                pass
            else:
                assert self.helper.config.lr_method == 'fix-lr'
                save_path = f'../saved/benign_new/{self.helper.config.dataset}_{epoch}_{self.helper.config.lr_method}.pt'
                torch.save(log_dict, save_path)
                print(f'Model saved at {save_path}')

    def save_res(self, accs, asrs):
        log_dict = {
            'accs': accs,
            'asrs': asrs
        }
        atk_method = self.helper.config.attacker_method
        if self.helper.config.sample_method == ['random', 'fix']:
            file_name = f'{self.helper.config.dataset}/{self.helper.config.agg_method}_{atk_method}_r_{self.helper.config.num_adversaries}_{self.helper.config.poison_epochs}_ts{self.helper.config.trigger_size}.pkl'
        else:
            raise NotImplementedError
        save_path = os.path.join(f'../saved/res/{file_name}')
        f_save = open(save_path, 'wb')
        pickle.dump(log_dict, f_save)
        f_save.close()
        print(f'results saved at {save_path}')

    def test_once(self, poison = False):
        model = self.helper.global_model
        model.eval()
        with torch.no_grad():               
            data_source = self.helper.test_data
            total_loss = 0
            correct = 0
            num_data = 0.
            for batch_id, batch in enumerate(data_source):
                data, targets = batch
                data, targets = data.cuda(), targets.cuda()
                if poison:
                    data, targets = self.attacker.poison_input(data, targets, eval=True)
                output = model(data)
                total_loss += self.criterion(output, targets).item()
                pred = output.data.max(1)[1] 
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
                num_data += output.size(0) 
        acc = float(correct) / float(num_data)
        # loss = total_loss / float(num_data)
        loss = total_loss
        model.train()
        return loss, acc
    
    def train(self):
        print('Training')
        accs = []
        asrs = []
        self.local_asrs = {}
        for epoch in range(-2, self.helper.config.epochs):
            print (f"epoch: {epoch} / {self.helper.config.epochs - 1}")
            sampled_participants = self.sample_participants(epoch)
            print (f"sampled_participants: {sampled_participants}")
            weight_accumulator = self.train_once(epoch, sampled_participants)
            self.aggregator.agg(self.helper.global_model, weight_accumulator)
            loss, acc = self.test_once()
            bkd_loss, bkd_acc = self.test_once(poison = self.helper.config.is_poison)
            self.log_once(epoch, loss, acc, bkd_loss, bkd_acc)
            accs.append(acc)
            asrs.append(bkd_acc)
        if self.helper.config.is_poison:
            self.save_res(accs, asrs)

    def train_once(self, epoch, sampled_participants):
        weight_accumulator = self.create_weight_accumulator()
        global_model_copy = self.create_global_model_copy()
        first_adversary = self.contain_adversary(epoch, sampled_participants)
        if first_adversary >= 0:
            model = self.helper.local_model
            self.copy_params(model, global_model_copy)
            self.attacker.train_atk_model(model, self.helper.train_data[first_adversary])
        print (f"fisrt adversary: {first_adversary}")

        for participant_id in sampled_participants:
            model = self.helper.local_model
            self.copy_params(model, global_model_copy)
            model.train()
            if not self.if_adversary(epoch, participant_id, sampled_participants):
                self.train_benign(participant_id, model, epoch)
            else:
                print ("train_malicious")
                self.train_malicious(participant_id, model, epoch)
            
            weight_accumulator = self.update_weight_accumulator(model, weight_accumulator)
        return weight_accumulator

    def train_malicious(self, participant_id, model, epoch):
        lr = self.get_lr(epoch)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
            momentum=self.helper.config.momentum,
            weight_decay=self.helper.config.decay)
        for internal_epoch in range(self.helper.config.attacker_retrain_times):
            for inputs, labels in self.helper.train_data[participant_id]:
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = self.attacker.poison_input(inputs, labels)
                output = model(inputs)
                loss = self.attacker_criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
        def val_asr(model, dl):
            model.eval()
            self.attacker.atk_model.eval()
            ce_loss = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)
            correct = 0.
            num_data = 0.
            total_loss = 0.
            with torch.no_grad():
                for inputs, labels in dl:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    inputs, labels = self.attacker.poison_input(inputs, labels, True)
                    output = model(inputs)
                    loss = ce_loss(output, labels)
                    total_loss += loss
                    pred = output.data.max(1)[1] 
                    correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
                    num_data += output.size(0)
            asr = correct/num_data
            self.attacker.atk_model.train()
            model.train()
            return asr, total_loss
        asr, loss = val_asr(model, self.helper.train_data[participant_id])
        print (asr, loss)
        asr, loss = val_asr(model, self.helper.test_data)
        print (asr, loss)

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
            if self.helper.config.sample_method in ['random', 'fix']:
                for p in sampled_participants:
                    if p < self.helper.config.num_adversaries:
                        return p
            elif self.helper.config.sample_method == 'random_updates':
                if epoch in self.advs:
                    return self.advs[epoch][0]
            else:
                raise NotImplementedError
        return -1

    def if_adversary(self, epoch, participant_id, sampled_participants):
        if self.helper.config.is_poison and epoch < self.helper.config.poison_epochs and epoch >= 0:
            if self.helper.config.sample_method in ['random', 'fix'] and participant_id < self.helper.config.num_adversaries:
                return True 
            elif self.helper.config.sample_method == 'random_updates':
                if epoch in self.advs:
                    for idx in self.advs[epoch]:
                        if sampled_participants[idx] == participant_id:
                            return True
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
                #if self.helper.config.dataset == 'cifar10':
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
        if self.helper.config.sample_method in ['random', 'random_updates']:
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
    
    def copy_params(self, model, target_params_variables):
        for name, layer in model.named_parameters():
            layer.data = copy.deepcopy(target_params_variables[name])
        
        