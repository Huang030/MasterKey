import sys
sys.path.append("../")
import time
import wandb
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .attack_model import MNISTConditionalAutoencoder, ConditionalAutoencoder
from .utils import grad_mask_cv, apply_grad_mask, proj_lp

MNIST_DEFAULT_MEAN = [0.5]
MNIST_DEFAULT_STD = [0.5]
MNIST_MIN = -1
MNIST_MAX = 1
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_MIN  = ((np.array([0,0,0]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).min()
IMAGENET_MAX  = ((np.array([1,1,1]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).max()


class Attacker:
    def __init__(self, helper):
        self.helper = helper
        self.num_classes = self.helper.config.num_classes
        if self.helper.config.dataset == 'mnist':
            self.atk_model = MNISTConditionalAutoencoder()
        elif self.helper.config.dataset == 'fmnist':
            self.atk_model = MNISTConditionalAutoencoder()
        elif self.helper.config.dataset == 'cifar10':
            self.atk_model = ConditionalAutoencoder(n_classes=10, input_dim=32)
        elif self.helper.config.dataset == 'gtsrb':
            self.atk_model = ConditionalAutoencoder(n_classes=43, input_dim=32)
        elif self.helper.config.dataset == 'cifar100':
            self.atk_model = ConditionalAutoencoder(n_classes=100, input_dim=32)
        elif self.helper.config.dataset == 'tiny-imagenet':
            self.atk_model = ConditionalAutoencoder(n_classes=200, input_dim=64)
        else:
            raise NotImplementedError
        
        self.atk_model.cuda().eval()
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)
        self.atk_optimizer = optim.Adam(self.atk_model.parameters(), lr=self.helper.config.atk_model_lr)
        self.original_eps = self.helper.config.original_eps
        self.cur_training_eps = self.original_eps

    def sample_attack_labels(self, label, n_classes):
        label_cpu = label.cpu().numpy()
        neg_label = [np.random.choice([e for e in range(n_classes) if e != l], 1)[0] for l in label_cpu]
        neg_label = torch.tensor(np.array(neg_label)).cuda()
        return neg_label    

    def clip_inputs(self, x):
        if self.helper.config.dataset == 'mnist':
            return torch.clamp(x, -1.0, 1.0)
        elif self.helper.config.dataset == 'fmnist':
            return torch.clamp(x, -1.0, 1.0)
        elif self.helper.config.dataset == 'cifar10':
            return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
        elif self.helper.config.dataset == 'gtsrb':
            return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
        elif self.helper.config.dataset == 'cifar100':
            return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
        elif self.helper.config.dataset == 'tiny-imagenet':
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
                x = inputs.clone()
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
            atk_labels = torch.ones_like(labels[:bkd_num]) * target
        else:
            atk_labels = self.sample_attack_labels(labels[:bkd_num], self.num_classes)
        noise = self.atk_model(inputs[:bkd_num], atk_labels) * self.cur_training_eps
        inputs[:bkd_num] = self.clip_inputs(inputs[:bkd_num] + noise)
        labels[:bkd_num] = atk_labels
        return inputs, labels

class A3FL_MT:
    def __init__(self, helper):
        self.helper = helper
        # self.atk_targets = self.helper.config.num_classes
        self.atk_targets = self.helper.config.num_classes
        self.triggers, self.masks = [], []
        self.target = 0 # 初始warmup没有攻击默认目标为0
        self.trigger_size = 5
        self.trigger_lr = 0.01
        self.dm_adv_K = 1
        self.noise_loss_lambda = 0.01
        self.dm_adv_model_count = 1
        if self.helper.config.dataset in ['mnist', 'fmnist']:
            trigger_size = (1,1,28,28)
            mean = MNIST_DEFAULT_MEAN
            std = MNIST_DEFAULT_STD
        else:
            trigger_size = (1,3,32,32)
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        for i in range(self.atk_targets):
            trigger = torch.rand(trigger_size, requires_grad=False, device = 'cuda')
            for c in range(trigger.size(1)):  # 对于每个通道
                trigger[:, c] = (trigger[:, c] - mean[c]) / std[c]         
            mask = torch.zeros_like(trigger, device = 'cuda')
            mask[:, :, 2:2+self.trigger_size, 2:2+self.trigger_size] = 1
            self.triggers.append(trigger)
            self.masks.append(mask)
    
    def choose_target(self, epoch):
        self.target = epoch % self.atk_targets

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

    def val_asr(self, model, dl, t, m):
        ce_loss = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)
        correct = 0.
        num_data = 0.
        total_loss = 0.
        with torch.no_grad():
            for inputs, labels in dl:
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs = t*m +(1-m)*inputs
                labels[:] = self.target
                output = model(inputs)
                loss = ce_loss(output, labels)
                total_loss += loss
                pred = output.data.max(1)[1] 
                correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
                num_data += output.size(0)
        asr = correct/num_data
        return asr, total_loss

    def search_trigger(self, model, dl, epoch):
        K = self.helper.config.a3fl_k 
        model.eval()
        # asr, _ = self.val_asr(model, dl, self.triggers[self.target], self.masks[self.target])
        # print (f"Target {self.target}")
        self.choose_target(epoch)
        print (f"Current Target: {self.target}")
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
                if self.helper.config.dataset in ['mnist', 'fmnist']:
                    t = torch.clamp(t, min = MNIST_MIN, max = MNIST_MAX)
                else:
                    t = torch.clamp(t, min = IMAGENET_MIN, max = IMAGENET_MAX)
                            
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
        bkd_num = inputs.shape[0] if eval else int(self.helper.config.bkd_ratio * inputs.shape[0])
        if eval and target != None:
            atk_labels = torch.ones_like(labels[:bkd_num]) * target
        elif eval:
            atk_labels = self.sample_attack_labels(labels[:bkd_num], self.atk_targets)
        else:
            atk_labels = self.sample_attack_labels(labels[:bkd_num], self.atk_targets)

        triggers, masks = [], []
        for label in atk_labels:
            target = int(label)
            triggers.append(self.triggers[target])
            masks.append(self.masks[target])
        triggers = torch.concat(triggers, dim=0)
        masks = torch.concat(masks, dim=0)

        inputs[:bkd_num] = triggers * masks + inputs[:bkd_num] * (1 - masks)
        labels[:bkd_num] = atk_labels

        return inputs, labels 

class CerP_MT:
    def __init__(self, helper):
        self.helper = helper
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)
        self.atk_targets = self.helper.config.num_classes
        triggers, self.masks = [], []
        self.target = 0 # 初始warmup没有攻击默认目标为0
        self.trigger_size = 5
        if self.helper.config.dataset in ['mnist', 'fmnist']:
            trigger_size = (1,1,28,28)
            mean = MNIST_DEFAULT_MEAN
            std = MNIST_DEFAULT_STD
        else:
            trigger_size = (1,3,32,32)
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        for i in range(self.atk_targets):
            trigger = torch.rand(trigger_size, requires_grad=False, device = 'cuda')
            for c in range(trigger.size(1)):  # 对于每个通道
                trigger[:, c] = (trigger[:, c] - mean[c]) / std[c] 
            mask = torch.zeros_like(trigger)
            mask[:, :, 2:2+self.trigger_size, 2:2+self.trigger_size] = 1
            mask = mask.cuda()
            trigger = trigger * mask
            triggers.append(trigger)
            self.masks.append(mask)
        self.intinal_triggers = triggers
        self.pre_triggers = copy.deepcopy(self.intinal_triggers)

    def choose_target(self, epoch):
        self.target = epoch % self.atk_targets

    def val_asr(self, model, dl, t, m):
        ce_loss = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)
        correct = 0.
        num_data = 0.
        total_loss = 0.
        with torch.no_grad():
            for inputs, labels in dl:
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs = t*m +(1-m)*inputs
                labels[:] = self.target
                output = model(inputs)
                loss = ce_loss(output, labels)
                total_loss += loss
                pred = output.data.max(1)[1] 
                correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
                num_data += output.size(0)
        asr = correct/num_data
        return asr, total_loss
        
    def search_trigger(self, model, dl, epoch):
        K = self.helper.config.cerp_k
        model.eval()
        self.choose_target(epoch)
        print (f"Current Target: {self.target}")

        init = False
        pre_trigger = self.pre_triggers[self.target].clone().detach().cuda()
        aa = copy.deepcopy(self.intinal_triggers[self.target]).cuda()
        m = self.masks[self.target]

        for e in range(K):
            for batch_id, (datas, labels) in enumerate(dl):
                x, y = datas.cuda(), labels.cuda()
                y_target = torch.LongTensor(y.size()).fill_(int(self.target))
                y_target = y_target.cuda()
                if not init:
                    noise = copy.deepcopy(pre_trigger)
                    noise.requires_grad_(True)
                    init = True

                x_noise = noise*m +(1-m)*x
                output = model((x_noise).float())
                classloss = nn.functional.cross_entropy(output, y_target)
                loss = classloss
                model.zero_grad()
                if noise.grad:
                    noise.grad.fill_(0)
                loss.backward(retain_graph=True)

                noise = noise - noise.grad * 0.1
                noise = noise * m

                delta_noise = noise - aa
                noise = aa + proj_lp(delta_noise, 10, 2)

                noise = noise.detach_()

                if self.helper.config.dataset in ['mnist', 'fmnist']:
                    noise = torch.clamp(noise, min = MNIST_MIN, max = MNIST_MAX)
                else:
                    noise = torch.clamp(noise, min = IMAGENET_MIN, max = IMAGENET_MAX)

                noise.requires_grad_(True)

        noise = noise.detach_()
        self.pre_triggers[self.target] = noise

    def sample_attack_labels(self, label, n_classes):
        label_cpu = label.cpu().numpy()
        if (n_classes != 1):
            neg_label = [np.random.choice([e for e in range(n_classes) if e != l], 1)[0] for l in label_cpu]
            neg_label = torch.tensor(np.array(neg_label)).cuda()
        else:
            neg_label = torch.zeros_like(label)
        return neg_label  

    def poison_input(self, inputs, labels, eval=False, target=None):
        bkd_num = inputs.shape[0] if eval else int(self.helper.config.bkd_ratio * inputs.shape[0])
        if eval and target != None:
            atk_labels = torch.ones_like(labels[:bkd_num]) * target
        elif eval:
            atk_labels = self.sample_attack_labels(labels[:bkd_num], self.atk_targets)
        else:
            atk_labels = self.sample_attack_labels(labels[:bkd_num], self.atk_targets)

        triggers, masks = [], []
        for label in atk_labels:
            target = int(label)
            triggers.append(self.pre_triggers[target])
            masks.append(self.masks[target])
        triggers = torch.concat(triggers, dim=0)
        masks = torch.concat(masks, dim=0)

        inputs[:bkd_num] = triggers * masks + inputs[:bkd_num] * (1 - masks)
        labels[:bkd_num] = atk_labels

        return inputs, labels 

class Neu_MT:
    def __init__(self, helper):
        self.helper = helper
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)
        self.atk_targets = self.helper.config.num_classes
        self.triggers, self.masks = [], []
        self.trigger_size = 5
        if self.helper.config.dataset in ['mnist', 'fmnist']:
            trigger_size = (1,1,28,28)
            mean = MNIST_DEFAULT_MEAN
            std = MNIST_DEFAULT_STD
        else:
            trigger_size = (1,3,32,32)
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        for i in range(self.atk_targets):
            trigger = torch.rand(trigger_size, requires_grad=False, device = 'cuda')
            for c in range(trigger.size(1)):  # 对于每个通道
                trigger[:, c] = (trigger[:, c] - mean[c]) / std[c] 
            mask = torch.zeros_like(trigger)
            mask[:, :, 2:2+self.trigger_size, 2:2+self.trigger_size] = 1
            mask = mask.cuda()
            self.triggers.append(trigger)
            self.masks.append(mask)
        num_clean_data = 30
        subset_data_chunks = random.sample(range(self.helper.config.num_total_participants)[1:], num_clean_data)
        self.sampled_data = [self.helper.train_data[pos] for pos in subset_data_chunks]
    
    def grad_mask(self, model, dl):
        mask_grad_list = grad_mask_cv(model, self.sampled_data, self.criterion, ratio=0.95)
        print ("Finish compute mask grad list.")
        return mask_grad_list

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

class PGD_MT:
    def __init__(self, helper):
        self.helper = helper
        self.atk_targets = self.helper.config.num_classes
        self.triggers, self.masks = [], []
        self.trigger_size = 5
        if self.helper.config.dataset in ['mnist', 'fmnist']:
            trigger_size = (1,1,28,28)
            mean = MNIST_DEFAULT_MEAN
            std = MNIST_DEFAULT_STD
        else:
            trigger_size = (1,3,32,32)
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        for i in range(self.atk_targets):
            trigger = torch.rand(trigger_size, requires_grad=False, device = 'cuda')
            for c in range(trigger.size(1)):  # 对于每个通道
                trigger[:, c] = (trigger[:, c] - mean[c]) / std[c] 
            mask = torch.zeros_like(trigger)
            mask[:, :, 2:2+self.trigger_size, 2:2+self.trigger_size] = 1
            mask = mask.cuda()
            self.triggers.append(trigger)
            self.masks.append(mask)
        self.project_frequency = 10
        self.eps = 1

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

# 弃用
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


    def __init__(self, helper):
        self.helper = helper
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)
        self.atk_targets = self.helper.config.num_classes
        triggers, self.masks = [], []
        self.target = 0 # 初始warmup没有攻击默认目标为0
        self.trigger_size = 5
        if self.helper.config.dataset in ['mnist', 'fmnist']:
            trigger_size = (1,1,28,28)
            mean = MNIST_DEFAULT_MEAN
            std = MNIST_DEFAULT_STD
        else:
            trigger_size = (1,3,32,32)
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        for i in range(self.atk_targets):
            trigger = torch.rand(trigger_size, requires_grad=False, device = 'cuda')
            for c in range(trigger.size(1)):  # 对于每个通道
                trigger[:, c] = (trigger[:, c] - mean[c]) / std[c] 
            mask = torch.zeros_like(trigger)
            mask[:, :, 2:2+self.trigger_size, 2:2+self.trigger_size] = 1
            mask = mask.cuda()
            trigger = trigger * mask
            triggers.append(trigger)
            self.masks.append(mask)
        self.intinal_triggers = triggers
        self.pre_triggers = copy.deepcopy(self.intinal_triggers)

    def choose_target(self, asr):
        if asr >= 0.85:
            self.target += 1
            self.target = self.target % self.atk_targets

    def val_asr(self, model, dl, t, m):
        ce_loss = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)
        correct = 0.
        num_data = 0.
        total_loss = 0.
        with torch.no_grad():
            for inputs, labels in dl:
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs = t*m +(1-m)*inputs
                labels[:] = self.target
                output = model(inputs)
                loss = ce_loss(output, labels)
                total_loss += loss
                pred = output.data.max(1)[1] 
                correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
                num_data += output.size(0)
        asr = correct/num_data
        return asr, total_loss
        
    def search_trigger(self, model, dl, epoch):
        model.eval()
        asr, _ = self.val_asr(model, dl, self.pre_triggers[self.target], self.masks[self.target])
        print (f"Target {self.target} ASR: {asr}")
        self.choose_target(asr)
        print (f"Current Target: {self.target}")

        init = False
        pre_trigger = self.pre_triggers[self.target].clone().detach().cuda()
        aa = copy.deepcopy(self.intinal_triggers[self.target]).cuda()
        m = self.masks[self.target]

        for e in range(1):
            for batch_id, (datas, labels) in enumerate(dl):
                x, y = datas.cuda(), labels.cuda()
                y_target = torch.LongTensor(y.size()).fill_(int(self.target))
                y_target = y_target.cuda()
                if not init:
                    noise = copy.deepcopy(pre_trigger)
                    noise.requires_grad_(True)
                    init = True

                x_noise = noise*m +(1-m)*x
                output = model((x_noise).float())
                classloss = nn.functional.cross_entropy(output, y_target)
                loss = classloss
                model.zero_grad()
                if noise.grad:
                    noise.grad.fill_(0)
                loss.backward(retain_graph=True)

                noise = noise - noise.grad * 0.1
                noise = noise * m

                delta_noise = noise - aa
                noise = aa + proj_lp(delta_noise, 10, 2)

                noise = noise.detach_()

                if self.helper.config.dataset in ['mnist', 'fmnist']:
                    noise = torch.clamp(noise, min = MNIST_MIN, max = MNIST_MAX)
                else:
                    noise = torch.clamp(noise, min = IMAGENET_MIN, max = IMAGENET_MAX)

                noise.requires_grad_(True)

        noise = noise.detach_()
        self.pre_triggers[self.target] = noise

    def sample_attack_labels(self, label, n_classes):
        label_cpu = label.cpu().numpy()
        if (n_classes != 1):
            neg_label = [np.random.choice([e for e in range(n_classes) if e != l], 1)[0] for l in label_cpu]
            neg_label = torch.tensor(np.array(neg_label)).cuda()
        else:
            neg_label = torch.zeros_like(label)
        return neg_label  

    def poison_input(self, inputs, labels, eval=False, target=None):
        bkd_num = inputs.shape[0] if eval else int(self.helper.config.bkd_ratio * inputs.shape[0])
        if eval and target != None:
            atk_labels = torch.ones_like(labels[:bkd_num]) * target
        elif eval:
            atk_labels = self.sample_attack_labels(labels[:bkd_num], self.atk_targets)
        else:
            atk_labels = torch.ones_like(labels[:bkd_num]) * self.target

        triggers, masks = [], []
        for label in atk_labels:
            target = int(label)
            triggers.append(self.pre_triggers[target])
            masks.append(self.masks[target])
        triggers = torch.concat(triggers, dim=0)
        masks = torch.concat(masks, dim=0)

        inputs[:bkd_num] = triggers * masks + inputs[:bkd_num] * (1 - masks)
        labels[:bkd_num] = atk_labels

        return inputs, labels 