import sys
sys.path.append("../")
import wandb
import torch
import torch.nn.functional as F
import torchvision
import random
import numpy as np
import copy
import os
from .attacker import Attacker, BaselineAttacker_1, BaselineAttacker_2
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
            attacks = {"MasterKey": Attacker, "PatchRound": BaselineAttacker_1, "PatchAll": BaselineAttacker_2}
            self.attacker = attacks[self.helper.config.attacker_method](self.helper)
        else:
            self.attacker = None
        self.save_attack_model = True
        self.setup_save_path()

        if self.helper.config.load_benign_model:
            model_path = f'../saved/benign_new/{self.helper.config.dataset}_{self.helper.config.poison_start_epoch}_{self.helper.config.lr_method}.pt'
            self.helper.global_model.load_state_dict(torch.load(model_path, map_location = 'cuda')['model'])
            loss,acc = self.test_once()
            print(f'Load benign model {model_path}, acc {acc:.3f}')
        return

    def setup_save_path(self):
        self.images_save_path = f'../saved/images/eps_{self.helper.config.eps}_atkepochs_{self.helper.config.atk_model_epochs}/'
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
        if (epoch + 1) % self.helper.config.save_every == 0:
            log_dict['model'] = self.helper.global_model.state_dict()
            if self.helper.config.is_poison:
                pass
            else:
                assert self.helper.config.lr_method == 'fix-lr'
                save_path = f'../saved/benign_new/{self.helper.config.dataset}_{epoch}_{self.helper.config.lr_method}.pt'
                torch.save(log_dict, save_path)
                print(f'Model saved at {save_path}')

    def test_once(self, poison = False, epoch = None, target = None):
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
                clean_img = data.clone()
                if poison:
                    if not target:
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
        print('Training')
        self.local_asrs = {}
        for epoch in range(-2, self.helper.config.epochs):
            print (f"Epoch: {epoch} / {self.helper.config.epochs - 1}")
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
        local_models = []
        local_weights = []
        local_model = self.helper.local_model
        global_model = self.helper.global_model

        first_adversary = self.contain_adversary(epoch, sampled_participants)
        print (f"fisrt adversary: {first_adversary}")
        if first_adversary >= 0:
            model = local_model
            self.copy_params(model, global_model)
            self.attacker.search_trigger(model, self.helper.train_data[first_adversary], epoch)

        for participant_id in sampled_participants:
            model = local_model
            self.copy_params(model, global_model)
            model.train()
            if not self.helper.config.is_poison or not self.if_adversary(epoch, participant_id, sampled_participants):
                self.train_benign(participant_id, model, epoch)
            else:
                self.train_malicious(participant_id, model, epoch)
            
            local_models.append(copy.deepcopy(model))
            local_weights.append(1.0)
        local_weights = [i/sum(local_weights) for i in local_weights]
        return local_models, local_weights

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
            model.train()
            return asr, total_loss
        # asr, loss = val_asr(model, self.helper.train_data[participant_id])
        # print (f"After Backdoor Injection  ===>asr: {asr}, loss: {loss}<===")

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
            if self.helper.config.sample_method == 'fix':
                for p in sampled_participants:
                    if p < self.helper.config.num_adversaries:
                        return p
            else:
                raise NotImplementedError
        return -1

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
        
        