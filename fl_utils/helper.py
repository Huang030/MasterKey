import sys
sys.path.append("../")
import os

import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset

import torchvision
from torchvision import datasets
from torchvision import datasets, transforms
from PIL import Image

from collections import defaultdict
import random
import numpy as np
from models.resnet import ResNet18
from models.mnist_cnn import NetC_MNIST

class Helper:
    def __init__(self, config):
        self.config = config
        self.config.data_folder = '~/data'
        self.local_model = None
        self.global_model = None
        self.setup_all()

    def setup_all(self):
        if self.config.dataset == 'mnist':
            self.load_mnist()
            self.load_cnn()
        elif self.config.dataset == 'cifar10':
            self.load_cifar10()
            self.load_resnet18()

    def load_cnn(self):
        self.local_model = NetC_MNIST()
        self.local_model.cuda()
        self.global_model = NetC_MNIST()
        self.global_model.cuda()

    def load_resnet18(self):
        self.local_model = ResNet18(num_classes = self.num_classes)
        self.local_model.cuda()
        self.global_model = ResNet18(num_classes = self.num_classes)
        self.global_model.cuda()
        
    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes):
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        return per_participant_list
    
    def get_train(self, indices):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
            num_workers=self.config.num_worker)
        return train_loader

    def get_test(self):

        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config.test_batch_size,
            shuffle=False,
            num_workers=self.config.num_worker)

        return test_loader

    def load_cifar10(self):
        self.num_classes = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.train_dataset = datasets.CIFAR10(
            self.config.data_folder, train=True, 
            download=True, transform=transform_train)
        self.test_dataset = datasets.CIFAR10(
            self.config.data_folder, train=False, transform=transform_test)
        
        indices_per_participant = self.sample_dirichlet_train_data(
            self.config.num_total_participants,
            alpha=self.config.dirichlet_alpha)
        
        train_loaders = [self.get_train(indices) 
            for pos, indices in indices_per_participant.items()]

        self.train_data = train_loaders
        self.test_data = self.get_test()
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_worker)
        
    def load_mnist(self):
        self.num_classes = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307), (0.3081)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307), (0.3081)),
        ])
        self.train_dataset = datasets.MNIST(
            self.config.data_folder, train=True, 
            download=True, transform=transform_train)
        self.test_dataset = datasets.MNIST(
            self.config.data_folder, train=False, transform=transform_test)
        
        indices_per_participant = self.sample_dirichlet_train_data(
            self.config.num_total_participants,
            alpha=self.config.dirichlet_alpha)
        
        train_loaders = [self.get_train(indices) 
            for pos, indices in indices_per_participant.items()]

        self.train_data = train_loaders
        self.test_data = self.get_test()

