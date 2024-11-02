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
import csv
import numpy as np
from models.resnet import ResNet18
from models.mnist_cnn import NetC_MNIST

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_MIN  = ((np.array([0,0,0]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).min()
IMAGENET_MAX  = ((np.array([1,1,1]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).max()


class GTSRB(Dataset):
    def __init__(self, data_root, train, transforms, min_width=0):
        super(GTSRB, self).__init__()
        if train:
            self.data_folder = os.path.join(data_root, "GTSRB/Train")
            self.images, self.labels = self._get_data_train_list(min_width=min_width)
            print(f'Loading GTSRB Train greater than {min_width} width. Loaded {len(self.images)} images.')
        else:
            self.data_folder = os.path.join(data_root, "GTSRB/Test")
            self.images, self.labels = self._get_data_test_list(min_width)
            print(f'Loading GTSRB Test greater than {min_width} width. Loaded {len(self.images)} images.')

        self.transforms = transforms

    def _get_data_train_list(self, min_width=0):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                if int(row[1]) >= min_width:
                    images.append(prefix + row[0])
                    labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self, min_width=0):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            if int(row[1]) >= min_width: #only load images if more than certain width
                images.append(self.data_folder + "/" + row[0])
                labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        return image, label
    

class Helper:
    def __init__(self, config):
        self.config = config
        self.config.data_folder = '../data/'
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
        elif self.config.dataset == 'gtsrb':
            self.load_gtsrb()
            self.load_resnet18()
        elif self.config.dataset == 'tiny-imagenet':
            raise NotImplementedError    

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
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes):
            random.shuffle(cifar_classes[n])
            class_size = len(cifar_classes[n])
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
        
    def load_gtsrb(self):
        self.num_classes = 43
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
        self.train_dataset = GTSRB(
            self.config.data_folder, train=True,
            transforms=transform_train, min_width=0)
        self.test_dataset = GTSRB(
            self.config.data_folder, train=False,
            transforms=transform_train, min_width=0)
        
        indices_per_participant = self.sample_dirichlet_train_data(
            self.config.num_total_participants,
            alpha=self.config.dirichlet_alpha)
        
        train_loaders = [self.get_train(indices) 
            for pos, indices in indices_per_participant.items()]

        self.train_data = train_loaders
        self.test_data = self.get_test()