import sys
sys.path.append("../")
import time
import wandb
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .attack_model import MNISTConditionalAutoencoder, ConditionalAutoencoder

IMAGENET_DEFAULT_MEAN = (0.4914, 0.4822, 0.4465)
IMAGENET_DEFAULT_STD = (0.2023, 0.1994, 0.2010)
IMAGENET_MIN  = ((np.array([0,0,0]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).min()
IMAGENET_MAX  = ((np.array([1,1,1]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).max()
# -2.4 2.8

class Attacker:
    def __init__(self, helper):
        self.helper = helper
        if self.helper.config.dataset == 'mnist':
            self.atk_model = MNISTConditionalAutoencoder()
            self.num_classes = 10
        elif self.helper.config.dataset == 'cifar10':
            self.atk_model = ConditionalAutoencoder(n_classes=10, input_dim=32)
            self.num_classes = 10
        else:
            raise NotImplementedError
        self.atk_model.cuda()
        self.atk_model.eval()
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)
        self.atk_optimizer = optim.Adam(self.atk_model.parameters(), lr=self.helper.config.atk_model_lr)

        # # temp parameters
        # self.adv_nums = 5
        # self.adv_epochs = 10
        # self.adv_lr = 0.01

    def sample_attack_labels(self, label, n_classes):
        # return torch.ones_like(label) * 7
        label_cpu = label.cpu().numpy()
        neg_label = [np.random.choice([e for e in range(n_classes) if e != l], 1)[0] for l in label_cpu]
        neg_label = torch.tensor(np.array(neg_label)).cuda()
        return neg_label    

    def clip_inputs(self, x):
        if self.helper.config.dataset == 'mnist':
            return torch.clamp(x, -1.0, 1.0)
        elif self.helper.config.dataset == 'cifar10':
            return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)
        else:
            raise NotImplementedError

    # def get_adv_model(self, model, dl):
    #     self.atk_model.eval()
    #     adv_model = copy.deepcopy(model)
    #     adv_optimizer = optim.Adam(adv_model.parameters(), lr=self.adv_lr)
    #     adv_model.train()

    #     for _ in range(self.adv_epochs):
    #         for inputs, labels in dl:
    #             inputs, labels = inputs.cuda(), labels.cuda()
    #             atk_labels = self.sample_attack_labels(labels, self.num_classes)
    #             noise = self.atk_model(inputs, atk_labels) * self.helper.config.eps
    #             atk_inputs = self.clip_inputs(inputs + noise)
    #             atk_outputs = adv_model(atk_inputs)

    #             adv_loss = self.criterion(atk_outputs, labels)
    #             adv_optimizer.zero_grad()
    #             adv_loss.backward()
    #             adv_optimizer.step()
    #     adv_model.eval()
    #     return adv_model


    def train_atk_model(self, model, dl):
        self.atk_model.train()
        model.eval()

        for _ in range(self.helper.config.atk_model_epochs):
            # adv_models = [self.get_adv_model(model, dl).eval() for i in range(self.adv_nums)]
            for inputs, labels in dl:
                loss = []
                inputs, labels = inputs.cuda(), labels.cuda()
                atk_labels = self.sample_attack_labels(labels, self.num_classes)
                noise = self.atk_model(inputs, atk_labels) * self.helper.config.eps
                atk_inputs = self.clip_inputs(inputs + noise)
                atk_outputs = model(atk_inputs)
                
                atk_loss = self.criterion(atk_outputs, atk_labels) * 1
                loss.append(atk_loss)

                # for j in range(self.adv_nums):
                #     adv_model = adv_models[j]
                #     adv_outputs = adv_model(atk_inputs)
                #     adv_loss = self.criterion(adv_outputs, atk_labels)
                #     loss.append(adv_loss)

                loss = sum(loss)
                self.atk_optimizer.zero_grad()
                loss.backward()
                self.atk_optimizer.step()
        model.train()
        self.atk_model.eval()
        self.test_atk_model(model, dl)

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


    def poison_input(self, inputs, labels, eval=False):
        if eval:
            bkd_num = int(1.0 * inputs.shape[0])
        else:
            bkd_num = int(0.5 * inputs.shape[0])
        atk_labels = self.sample_attack_labels(labels[:bkd_num], 10)
        noise = self.atk_model(inputs[:bkd_num], atk_labels) * self.helper.config.eps
        inputs[:bkd_num] = self.clip_inputs(inputs[:bkd_num] + noise)
        labels[:bkd_num] = atk_labels
        return inputs, labels
    
