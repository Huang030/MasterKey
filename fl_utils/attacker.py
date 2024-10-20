import sys
sys.path.append("../")
import time
import wandb
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
        elif self.helper.config.dataset == 'cifar10':
            self.atk_model = ConditionalAutoencoder(n_classes=10, input_dim=32)
        else:
            raise NotImplementedError
        self.atk_model.cuda()
        self.atk_model.eval()
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)
        self.atk_optimizer = optim.Adam(self.atk_model.parameters(), lr=self.helper.config.atk_model_lr)

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
        else:
            raise NotImplementedError

    def train_atk_model(self, model, dl):
        self.atk_model.train()
        model.eval()
        ce_loss = torch.nn.CrossEntropyLoss()
        for _ in range(self.helper.config.atk_model_epochs):
            for inputs, labels in dl:
                inputs, labels = inputs.cuda(), labels.cuda()
                atk_labels = self.sample_attack_labels(labels, 10)
                noise = self.atk_model(inputs, atk_labels) * self.helper.config.eps
                atk_inputs = self.clip_inputs(inputs + noise)

                atk_outputs = model(atk_inputs)
                atk_loss = ce_loss(atk_outputs, atk_labels)
                self.atk_optimizer.zero_grad()
                atk_loss.backward()
                self.atk_optimizer.step()
        model.train()
        self.atk_model.eval()

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
    
