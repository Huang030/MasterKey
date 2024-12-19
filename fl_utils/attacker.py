import sys
sys.path.append("../")
import time
import wandb
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .attack_model import MNISTConditionalAutoencoder, ConditionalAutoencoder

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_MIN  = ((np.array([0,0,0]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).min()
IMAGENET_MAX  = ((np.array([1,1,1]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).max()


class Attacker:
    def __init__(self, helper):
        self.helper = helper
        if self.helper.config.dataset == 'mnist':
            self.atk_model = MNISTConditionalAutoencoder()
            self.num_classes = 10
        elif self.helper.config.dataset == 'cifar10':
            self.atk_model = ConditionalAutoencoder(n_classes=10, input_dim=32)
            self.num_classes = 10
        elif self.helper.config.dataset == 'gtsrb':
            self.atk_model = ConditionalAutoencoder(n_classes=32, input_dim=32)
            self.num_classes = 43
        else:
            raise NotImplementedError
        
        self.atk_model.cuda().eval()
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)
        self.atk_optimizer = optim.Adam(self.atk_model.parameters(), lr=self.helper.config.atk_model_lr)
        self.cur_training_eps = self.helper.config.eps

    def sample_attack_labels(self, label, n_classes):
        label_cpu = label.cpu().numpy()
        neg_label = [np.random.choice([e for e in range(n_classes) if e != l], 1)[0] for l in label_cpu]
        neg_label = torch.tensor(np.array(neg_label)).cuda()
        return neg_label    

    def clip_inputs(self, x):
        if self.helper.config.dataset == 'mnist':
            return torch.clamp(x, -1.0, 1.0)
        elif self.helper.config.dataset == 'cifar10':
            return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
        elif self.helper.config.dataset == 'gtsrb':
            return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
        else:
            raise NotImplementedError

    def search_trigger(self, model, dl, round):
        self.atk_model.train()
        model.eval()
        for _ in range(self.helper.config.atk_model_epochs):
            for inputs, labels in dl:
                loss = []
                inputs, labels = inputs.cuda(), labels.cuda()
                atk_inputs, atk_labels = self.poison_input(inputs, labels, eval=True)
                atk_outputs = model(atk_inputs)
                
                atk_loss = self.criterion(atk_outputs, atk_labels) * 1
                loss.append(atk_loss)

                loss = sum(loss)
                self.atk_optimizer.zero_grad()
                loss.backward()
                self.atk_optimizer.step()
        model.train()
        self.atk_model.eval()
        # self.test_atk_model(model, dl)

    def test_atk_model(self, model, dl):
        self.atk_model.eval()
        model.eval()
        with torch.no_grad():               
            data_source = dl
            total_loss = 0
            correct = 0
            num_data = 0.
            for batch_id, batch in enumerate(data_source):
                data, targets = batch
                data, targets = data.cuda(), targets.cuda()
                data, targets = self.poison_input(data, targets, eval=True)
                output = model(data)
                total_loss += self.criterion(output, targets).item()
                pred = output.data.max(1)[1] 
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
                num_data += output.size(0) 
        acc = float(correct) / float(num_data)
        loss = total_loss
        model.train()
        print (f"After Trigger Generating  ===>asr: {acc}, loss: {loss}<===")


    def poison_input(self, inputs, labels, eval=False, target=None):
        if eval:
            bkd_num = int(1.0 * inputs.shape[0])
        else:
            bkd_num = int(0.5 * inputs.shape[0])
        
        if target:
            atk_labels = torch.ones_like(inputs[:bkd_num]) * target
        else:
            atk_labels = self.sample_attack_labels(labels[:bkd_num], self.num_classes)
        noise = self.atk_model(inputs[:bkd_num], atk_labels) * self.cur_training_eps
        inputs[:bkd_num] = self.clip_inputs(inputs[:bkd_num] + noise)
        labels[:bkd_num] = atk_labels
        return inputs, labels


class BaselineAttacker_1:
    def __init__(self, helper):
        self.helper = helper
        self.atk_targets = self.helper.config.num_classes
        self.poison_epochs = self.helper.config.poison_epochs
        self.triggers, self.masks = [], []
        self.target = 0 # 初始warmup没有攻击默认目标为0
        self.trigger_size = 5
        self.trigger_lr = 0.01
        self.dm_adv_K = 1
        self.noise_loss_lambda = 0.01
        self.dm_adv_model_count = 1
        for i in range(self.atk_targets):
            trigger = torch.ones((1,3,32,32), requires_grad=False, device = 'cuda')*0.5
            mask = torch.zeros_like(trigger)
            mask[:, :, 2:2+self.trigger_size, 2:2+self.trigger_size] = 1
            mask = mask.cuda()
            self.triggers.append(trigger)
            self.masks.append(mask)

    def choose_target(self, epoch):
        # target = random.sample(range(self.atk_targets), 1)[0]
        round_per_target = self.poison_epochs // self.atk_targets
        target = (epoch // round_per_target) % (self.atk_targets + 1)
        return target

    def get_adv_model(self, model, dl, trigger, mask):
        adv_model = copy.deepcopy(model)
        adv_model.train()
        ce_loss = torch.nn.CrossEntropyLoss()
        adv_opt = torch.optim.SGD(adv_model.parameters(), lr = 0.01, momentum=0.9, weight_decay=5e-4)
        for _ in range(5):
            for inputs, labels in dl:
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs = trigger*mask +(1-mask)*inputs
                outputs = adv_model(inputs)
                loss = ce_loss(outputs, labels)
                adv_opt.zero_grad()
                loss.backward()
                adv_opt.step()

        sim_sum = 0.
        sim_count = 0.
        cos_loss = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
        for name in dict(adv_model.named_parameters()):
            if 'conv' in name:
                sim_count += 1
                sim_sum += cos_loss(dict(adv_model.named_parameters())[name].grad.reshape(-1),\
                                    dict(model.named_parameters())[name].grad.reshape(-1))
        return adv_model, sim_sum/sim_count

    def search_trigger(self, model, dl, epoch):
        self.target = self.choose_target(epoch)
        print (f"Current Target: {self.target}")
        K = 20
        model.eval()
        adv_models = []
        adv_ws = []
        ce_loss = torch.nn.CrossEntropyLoss()
        alpha = self.trigger_lr
        t = self.triggers[self.target].clone()
        m = self.masks[self.target].clone()
        count = 0
        for iter in range(K):
            if iter % self.dm_adv_K == 0 and iter != 0:
                if len(adv_models)>0:
                    for adv_model in adv_models:
                        del adv_model
                adv_models = []
                adv_ws = []
                for _ in range(1):
                    adv_model, adv_w = self.get_adv_model(model, dl, t,m) 
                    adv_models.append(adv_model)
                    adv_ws.append(adv_w)
            

            for inputs, labels in dl:
                count += 1
                t.requires_grad_()
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs = t*m +(1-m)*inputs
                labels[:] = self.target
                outputs = model(inputs) 
                loss = ce_loss(outputs, labels)
                
                if len(adv_models) > 0:
                    for am_idx in range(len(adv_models)):
                        adv_model = adv_models[am_idx]
                        adv_w = adv_ws[am_idx]
                        outputs = adv_model(inputs)
                        nm_loss = ce_loss(outputs, labels)
                        if loss == None:
                            loss = self.noise_loss_lambda*adv_w*nm_loss/self.dm_adv_model_count
                        else:
                            loss += self.noise_loss_lambda*adv_w*nm_loss/self.dm_adv_model_count
                if loss != None:
                    loss.backward()
                    new_t = t - alpha*t.grad.sign()
                    t = new_t.detach_()
                    t = torch.clamp(t, min = -2, max = 2)
                    t.requires_grad_()
        t = t.detach()
        self.triggers[self.target] = t
        self.masks[self.target] = m

    def sample_attack_labels(self, label, n_classes):
        label_cpu = label.cpu().numpy()
        if (n_classes != 1):
            neg_label = [np.random.choice([e for e in range(n_classes) if e != l], 1)[0] for l in label_cpu]
            neg_label = torch.tensor(np.array(neg_label)).cuda()
        else:
            neg_label = torch.zeros_like(label)
        return neg_label  

    def poison_input(self, inputs, labels, eval=False, target=None):
        if eval:
            bkd_num = inputs.shape[0]
            if target == None:
                atk_labels = self.sample_attack_labels(labels[:bkd_num], self.target + 1)
                triggers, masks = [], []
                for i in atk_labels:
                    target = int(i)
                    triggers.append(self.triggers[target])
                    masks.append(self.masks[target])
                triggers = torch.concat(triggers, dim=0)
                masks = torch.concat(masks, dim=0)
                inputs[:bkd_num] = triggers*masks + inputs[:bkd_num]*(1-masks)
                labels[:bkd_num] = atk_labels
            else:
                atk_labels = torch.ones_like(labels) * target
                inputs[:bkd_num] = self.triggers[target]*self.masks[target] + inputs[:bkd_num]*(1-self.masks[target])
                labels[:bkd_num] = atk_labels
        else:
            bkd_num = int(self.helper.config.bkd_ratio * inputs.shape[0])
            inputs[:bkd_num] = self.triggers[self.target]*self.masks[self.target] + inputs[:bkd_num]*(1-self.masks[self.target])
            labels[:bkd_num] = self.target
        return inputs, labels        

class BaselineAttacker_2:
    def __init__(self, helper):
        self.helper = helper
        self.atk_targets = self.helper.config.num_classes
        self.triggers, self.masks = [], []
        self.trigger_size = 5
        self.trigger_lr = 0.01
        self.dm_adv_K = 1
        self.noise_loss_lambda = 0.01
        self.dm_adv_model_count = 1
        for i in range(self.atk_targets):
            trigger = torch.ones((1,3,32,32), requires_grad=False, device = 'cuda')*0.5
            mask = torch.zeros_like(trigger)
            mask[:, :, 2:2+self.trigger_size, 2:2+self.trigger_size] = 1
            mask = mask.cuda()
            self.triggers.append(trigger)
            self.masks.append(mask)

    def get_adv_model(self, model, dl, trigger, mask):
        adv_model = copy.deepcopy(model)
        adv_model.train()
        ce_loss = torch.nn.CrossEntropyLoss()
        adv_opt = torch.optim.SGD(adv_model.parameters(), lr = 0.01, momentum=0.9, weight_decay=5e-4)
        for _ in range(5):
            for inputs, labels in dl:
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs = trigger*mask +(1-mask)*inputs
                outputs = adv_model(inputs)
                loss = ce_loss(outputs, labels)
                adv_opt.zero_grad()
                loss.backward()
                adv_opt.step()

        sim_sum = 0.
        sim_count = 0.
        cos_loss = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
        for name in dict(adv_model.named_parameters()):
            if 'conv' in name:
                sim_count += 1
                sim_sum += cos_loss(dict(adv_model.named_parameters())[name].grad.reshape(-1),\
                                    dict(model.named_parameters())[name].grad.reshape(-1))
        return adv_model, sim_sum/sim_count

    def search_trigger(self, model, dl, epoch):
        for target in range(self.atk_targets):
            K = 20
            model.eval()
            adv_models = []
            adv_ws = []
            ce_loss = torch.nn.CrossEntropyLoss()
            alpha = self.trigger_lr
            t = self.triggers[target].clone()
            m = self.masks[target].clone()
            count = 0
            for iter in range(K):
                if iter % self.dm_adv_K == 0 and iter != 0:
                    if len(adv_models)>0:
                        for adv_model in adv_models:
                            del adv_model
                    adv_models = []
                    adv_ws = []
                    for _ in range(1):
                        adv_model, adv_w = self.get_adv_model(model, dl, t,m) 
                        adv_models.append(adv_model)
                        adv_ws.append(adv_w)
                

                for inputs, labels in dl:
                    count += 1
                    t.requires_grad_()
                    inputs, labels = inputs.cuda(), labels.cuda()
                    inputs = t*m +(1-m)*inputs
                    labels[:] = target
                    outputs = model(inputs) 
                    loss = ce_loss(outputs, labels)
                    
                    if len(adv_models) > 0:
                        for am_idx in range(len(adv_models)):
                            adv_model = adv_models[am_idx]
                            adv_w = adv_ws[am_idx]
                            outputs = adv_model(inputs)
                            nm_loss = ce_loss(outputs, labels)
                            if loss == None:
                                loss = self.noise_loss_lambda*adv_w*nm_loss/self.dm_adv_model_count
                            else:
                                loss += self.noise_loss_lambda*adv_w*nm_loss/self.dm_adv_model_count
                    if loss != None:
                        loss.backward()
                        new_t = t - alpha*t.grad.sign()
                        t = new_t.detach_()
                        t = torch.clamp(t, min = -2, max = 2)
                        t.requires_grad_()
            t = t.detach()
            self.triggers[target] = t
            self.masks[target] = m

    def sample_attack_labels(self, label, n_classes):
        label_cpu = label.cpu().numpy()
        if (n_classes != 1):
            neg_label = [np.random.choice([e for e in range(n_classes) if e != l], 1)[0] for l in label_cpu]
            neg_label = torch.tensor(np.array(neg_label)).cuda()
        else:
            neg_label = torch.zeros_like(label)
        return neg_label  

    def poison_input(self, inputs, labels, eval=False, target=None):
        if eval:
            bkd_num = inputs.shape[0]
            if target == None:
                atk_labels = self.sample_attack_labels(labels[:bkd_num], self.atk_targets)
                triggers, masks = [], []
                for i in atk_labels:
                    target = int(i)
                    triggers.append(self.triggers[target])
                    masks.append(self.masks[target])
                triggers = torch.concat(triggers, dim=0)
                masks = torch.concat(masks, dim=0)
                inputs[:bkd_num] = triggers*masks + inputs[:bkd_num]*(1-masks)
                labels[:bkd_num] = atk_labels
            else:
                atk_labels = torch.ones_like(labels) * target
                inputs[:bkd_num] = self.triggers[target]*self.masks[target] + inputs[:bkd_num]*(1-self.masks[target])
                labels[:bkd_num] = atk_labels
        else:
            bkd_num = int(self.helper.config.bkd_ratio * inputs.shape[0])
            atk_labels = self.sample_attack_labels(labels[:bkd_num], self.atk_targets)
            triggers, masks = [], []
            for i in atk_labels:
                target = int(i)
                triggers.append(self.triggers[target])
                masks.append(self.masks[target])
            triggers = torch.concat(triggers, dim=0)
            masks = torch.concat(masks, dim=0)
            inputs[:bkd_num] = triggers*masks + inputs[:bkd_num]*(1-masks)
            labels[:bkd_num] = atk_labels
        return inputs, labels        
