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
from typing import Iterable, Optional

# Neu
def grad_mask_cv(model, dataset_clearn, criterion, ratio=0.5):
    """Generate a gradient mask based on the given dataset"""
    aggregate_all_layer = 1
    model.train()
    model.zero_grad()

    for participant_id in range(len(dataset_clearn)):

        train_data = dataset_clearn[participant_id]

        for inputs, labels in train_data:
            inputs, labels = inputs.cuda(), labels.cuda()

            output = model(inputs)

            loss = criterion(output, labels)
            loss.backward(retain_graph=True)

    mask_grad_list = []
    if aggregate_all_layer == 1:
        grad_list = []
        grad_abs_sum_list = []
        k_layer = 0
        for _, parms in model.named_parameters():
            if parms.requires_grad:
                grad_list.append(parms.grad.abs().view(-1))

                grad_abs_sum_list.append(parms.grad.abs().view(-1).sum().item())

                k_layer += 1

        grad_list = torch.cat(grad_list).cuda()
        _, indices = torch.topk(-1*grad_list, int(len(grad_list)*ratio))
        mask_flat_all_layer = torch.zeros(len(grad_list)).cuda()
        mask_flat_all_layer[indices] = 1.0

        count = 0
        percentage_mask_list = []
        k_layer = 0
        grad_abs_percentage_list = []
        for _, parms in model.named_parameters():
            if parms.requires_grad:
                gradients_length = len(parms.grad.abs().view(-1))

                mask_flat = mask_flat_all_layer[count:count + gradients_length ].cuda()
                mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())

                count += gradients_length

                percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0

                percentage_mask_list.append(percentage_mask1)

                grad_abs_percentage_list.append(grad_abs_sum_list[k_layer]/np.sum(grad_abs_sum_list))

                k_layer += 1
    else:
        grad_abs_percentage_list = []
        grad_res = []
        l2_norm_list = []
        sum_grad_layer = 0.0
        for _, parms in model.named_parameters():
            if parms.requires_grad:
                grad_res.append(parms.grad.view(-1))
                l2_norm_l = torch.norm(parms.grad.view(-1).clone().detach().cuda())/float(len(parms.grad.view(-1)))
                l2_norm_list.append(l2_norm_l)
                sum_grad_layer += l2_norm_l.item()

        grad_flat = torch.cat(grad_res)

        percentage_mask_list = []
        k_layer = 0
        for _, parms in model.named_parameters():
            if parms.requires_grad:
                gradients = parms.grad.abs().view(-1)
                gradients_length = len(gradients)
                if ratio == 1.0:
                    _, indices = torch.topk(-1*gradients, int(gradients_length*1.0))
                else:

                    ratio_tmp = 1 - l2_norm_list[k_layer].item() / sum_grad_layer
                    _, indices = torch.topk(-1*gradients, int(gradients_length*ratio))

                mask_flat = torch.zeros(gradients_length)
                mask_flat[indices.cpu()] = 1.0
                mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())

                percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0

                percentage_mask_list.append(percentage_mask1)


                k_layer += 1

    model.zero_grad()
    return mask_grad_list

def apply_grad_mask(model, mask_grad_list):
    mask_grad_list_copy = iter(mask_grad_list)
    for name, parms in model.named_parameters():
        if parms.requires_grad:
            parms.grad = parms.grad * next(mask_grad_list_copy)


# CerP
def proj_lp(v, xi, p):
    # Project on the lp ball centered at 0 and of radius xi
    # SUPPORTS only p = 2 and p = Inf for now
    if p == 2:
        v = v * min(1, xi / torch.norm(v))
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
        raise ValueError('Values of p different from 2 and Inf are currently not supported...')
    return v

def model_dist_norm_var(model, target_params_variables, norm=2):
    size = 0
    for name, layer in model.named_parameters():
        size += layer.view(-1).shape[0]
    sum_var = torch.FloatTensor(size).fill_(0)
    sum_var = sum_var.cuda()
    size = 0
    for name, layer in model.named_parameters():
        sum_var[size:size + layer.view(-1).shape[0]] = (
                layer - target_params_variables[name]).view(-1)
        size += layer.view(-1).shape[0]

    return torch.norm(sum_var, norm)


# PGD
def parameters_to_vector(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
    r"""Flatten an iterable of parameters into a single vector.

    Args:
        parameters (Iterable[Tensor]): an iterable of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    # Flag for the device where the parameter is located
    param_device = None

    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        vec.append(param.view(-1))
    return torch.cat(vec)


def vector_to_parameters(vec: torch.Tensor, parameters: Iterable[torch.Tensor]) -> None:
    r"""Copy slices of a vector into an iterable of parameters.

    Args:
        vec (Tensor): a single vector representing the parameters of a model.
        parameters (Iterable[Tensor]): an iterable of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError(f'expected torch.Tensor, but got: {torch.typename(vec)}')
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param


def _check_param_device(param: torch.Tensor, old_param_device: Optional[int]) -> int:
    r"""Check if the parameters are located on the same device.

    Currently, the conversion between model parameters and single vector form is not supported
    for multiple allocations, e.g. parameters in different GPUs/PrivateUse1s, or mixture of CPU/GPU/PrivateUse1.

    Args:
        param ([Tensor]): a Tensor of a parameter of a model
        old_param_device (int): the device where the first parameter of a
                                model is allocated.

    Returns:
        old_param_device (int): report device for the first time
    """
    # Meet the first parameter
    support_device_types = ["cuda", torch._C._get_privateuse1_backend_name()]
    if old_param_device is None:
        old_param_device = param.get_device() if param.device.type in support_device_types else -1
    else:
        warn = False
        if param.device.type in support_device_types:  # Check if in same GPU/PrivateUse1
            warn = (param.get_device() != old_param_device)
        else:  # Check if in CPU
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Found two parameters on different devices, '
                            'this is currently not supported.')
    return old_param_device



from tkinter.messagebox import NO
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import copy
import math
# from skimage import io

# -*- coding = utf-8 -*-
import cv2
import torch
import numpy as np

def add_trigger(args, image, test=False):
    pixel_max = max(1,torch.max(image))
    if args.attack == 'dba' and test == False:
        if args.dba_class == 0:
            # image[:, args.triggerY + 0:args.triggerY + 2, args.triggerX + 0:args.triggerX + size] = pixel_max
            image[:, args.triggerY + 0:args.triggerY + 2, args.triggerX + 0:args.triggerX + 2] = pixel_max
        elif args.dba_class == 1:
            # image[:, args.triggerY + 0:args.triggerY + 2, args.triggerX+size+gap:args.triggerX +size+gap+size] = pixel_max
            image[:, args.triggerY + 0:args.triggerY + 2, args.triggerX + 2:args.triggerX + 5] = pixel_max
        elif args.dba_class == 2:
            # image[:, args.triggerY + 2+gap:args.triggerY + 2+gap+2, args.triggerX + 0:args.triggerX + size] = pixel_max
            image[:, args.triggerY + 2:args.triggerY + 5, args.triggerX + 0:args.triggerX + 2] = pixel_max
        elif args.dba_class == 3:
            # image[:, args.triggerY + 2+gap:args.triggerY + 2+gap+2, args.triggerX +size+gap:args.triggerX +size+gap+size] = pixel_max
            image[:, args.triggerY + 2:args.triggerY + 5, args.triggerX + 2:args.triggerX + 5] = pixel_max
        args.save_img(image)
        return image
    if args.attack == 'dba' and test == True:
        image[:, args.triggerY + 0:args.triggerY + 2, args.triggerX + 0:args.triggerX + 2] = pixel_max
        image[:, args.triggerY + 0:args.triggerY + 2, args.triggerX + 2:args.triggerX + 5] = pixel_max
        image[:, args.triggerY + 2:args.triggerY + 5, args.triggerX + 0:args.triggerX + 2] = pixel_max
        image[:, args.triggerY + 2:args.triggerY + 5, args.triggerX + 2:args.triggerX + 5] = pixel_max

        return image
    if args.trigger == 'square':
        pixel_max = torch.max(image) if torch.max(image) > 1 else 1

        if args.dataset == 'cifar':
            pixel_max = 1
        image[:, args.triggerY:args.triggerY + 5, args.triggerX:args.triggerX + 5] = pixel_max
    elif args.trigger == 'pattern':
        pixel_max = torch.max(image) if torch.max(image) > 1 else 1
        image[:, args.triggerY + 0, args.triggerX + 0] = pixel_max
        image[:, args.triggerY + 1, args.triggerX + 1] = pixel_max
        image[:, args.triggerY - 1, args.triggerX + 1] = pixel_max
        image[:, args.triggerY + 1, args.triggerX - 1] = pixel_max
    elif args.trigger == 'watermark':
        if args.watermark is None:
            args.watermark = cv2.imread('./utils/watermark.png', cv2.IMREAD_GRAYSCALE)
            args.watermark = cv2.bitwise_not(args.watermark)
            args.watermark = cv2.resize(args.watermark, dsize=image[0].shape, interpolation=cv2.INTER_CUBIC)
            pixel_max = np.max(args.watermark)
            args.watermark = args.watermark.astype(np.float64) / pixel_max
            # cifar [0,1] else max>1
            pixel_max_dataset = torch.max(image).item() if torch.max(image).item() > 1 else 1
            args.watermark *= pixel_max_dataset
        max_pixel = max(np.max(args.watermark), torch.max(image))
        image += args.watermark
        image[image > max_pixel] = max_pixel
    elif args.trigger == 'apple':
        if args.apple is None:
            args.apple = cv2.imread('./utils/apple.png', cv2.IMREAD_GRAYSCALE)
            args.apple = cv2.bitwise_not(args.apple)
            args.apple = cv2.resize(args.apple, dsize=image[0].shape, interpolation=cv2.INTER_CUBIC)
            pixel_max = np.max(args.apple)
            args.apple = args.apple.astype(np.float64) / pixel_max
            # cifar [0,1] else max>1
            pixel_max_dataset = torch.max(image).item() if torch.max(image).item() > 1 else 1
            args.apple *= pixel_max_dataset
        max_pixel = max(np.max(args.apple), torch.max(image))
        image += args.apple
        image[image > max_pixel] = max_pixel
    elif args.trigger == 'hallokitty':
        if args.hallokitty is None:
            args.hallokitty = cv2.imread('./utils/halloKitty.png')
            pixel_max = np.max(args.hallokitty)
            args.hallokitty = args.hallokitty.astype(np.float64) / pixel_max
            args.hallokitty = torch.from_numpy(args.hallokitty)
            # cifar [0,1] else max>1
            pixel_max_dataset = torch.max(image).item() if torch.max(image).item() > 1 else 1
            args.hallokitty *= pixel_max_dataset
        image = args.hallokitty * 0.5 + image * 0.5
        max_pixel = max(torch.max(args.hallokitty), torch.max(image))
        image[image > max_pixel] = max_pixel
    # save the most recent backdoor image in test dataset
    # args.save_img(image)
    return image


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, helper, dataset=None, idxs=None):
        self.helper = helper
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(
            dataset, idxs), batch_size=self.helper.config.batch_size, shuffle=True)

            
    def get_PLR(self, net):
        # get penultimate layer representations from root dataset
        # return:
        # penultimate layer representations of images in root dataset
        features_list = []
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.cuda(), labels.cuda()
            net.zero_grad()
            features = net.features(images)
            features_list.append(features)
        features_list = torch.concat(features_list, dim=0)
        return features_list


def calculate_sparsities(dense_ratio, params, tabu=[], distribution="ERK"):
    spasities = {}
    if distribution == "uniform":
        for name in params:
            if name not in tabu:
                spasities[name] = 1 - dense_ratio
            else:
                spasities[name] = 0
    elif distribution == "ERK":
        print('initialize by ERK')
        total_params = 0
        for name in params:
            total_params += params[name].numel()
        is_epsilon_valid = False
        # # The following loop will terminate worst case when all masks are in the
        # custom_sparsity_map. This should probably never happen though, since once
        # we have a single variable or more with the same constant, we have a valid
        # epsilon. Note that for each iteration we add at least one variable to the
        # custom_sparsity_map and therefore this while loop should terminate.
        dense_layers = set()

        density = dense_ratio
        while not is_epsilon_valid:
            # We will start with all layers and try to find right epsilon. However if
            # any probablity exceeds 1, we will make that layer dense and repeat the
            # process (finding epsilon) with the non-dense layers.
            # We want the total number of connections to be the same. Let say we have
            # for layers with N_1, ..., N_4 parameters each. Let say after some
            # iterations probability of some dense layers (3, 4) exceeded 1 and
            # therefore we added them to the dense_layers set. Those layers will not
            # scale with erdos_renyi, however we need to count them so that target
            # paratemeter count is achieved. See below.
            # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
            #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
            # eps * (p_1 * N_1 + p_2 * N_2) =
            #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
            # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

            divisor = 0
            rhs = 0
            raw_probabilities = {}
            for name in params:
                if name in tabu or "running" in name or "track" in name :
                    dense_layers.add(name)
                n_param = np.prod(params[name].shape)
                n_zeros = n_param * (1 - density)
                n_ones = n_param * density

                if name in dense_layers:
                    rhs -= n_zeros
                else:
                    rhs += n_ones
                    raw_probabilities[name] = (
                                                      np.sum(params[name].shape) / np.prod(params[name].shape)
                                              ) ** 1
                    divisor += raw_probabilities[name] * n_param
            epsilon = rhs / divisor
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid = False
                for mask_name, mask_raw_prob in raw_probabilities.items():
                    if mask_raw_prob == max_prob:
                        # print(f"Sparsity of var:{mask_name} had to be set to 0.")
                        dense_layers.add(mask_name)
            else:
                is_epsilon_valid = True

        # With the valid epsilon, we can set sparsities of the remaning layers.
        for name in params:
            if name in dense_layers:
                spasities[name] = 0
            else:
                spasities[name] = (1 - epsilon * raw_probabilities[name])
    return spasities

def init_masks(params, sparsities):
    masks = {}
    for name in params:
        masks[name] = torch.zeros_like(params[name])
        dense_numel = int((1 - sparsities[name]) * torch.numel(masks[name]))
        if dense_numel > 0:
            temp = masks[name].view(-1)
            perm = torch.randperm(len(temp))
            perm = perm[:dense_numel]
            temp[perm] = 1
        masks[name] = masks[name].to("cpu")
    return masks


class Agent_s():
    def __init__(self, id, helper, mask=None):
        self.id = id
        self.helper = helper
        self.error = 0
        self.mask = copy.deepcopy(mask)
        self.num_remove= None
        self.train_loader = self.helper.train_data[id]
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)


    def screen_gradients(self, model):
        model.train()
        # # # train and update
        criterion = nn.CrossEntropyLoss()
        gradient = {name: 0 for name, param in model.named_parameters()}
        # # sample 10 batch  of data
        batch_num = 0
        for _, (x, labels) in enumerate(self.train_loader):
            batch_num+=1
            model.zero_grad()
            x, labels = x.cuda(), labels.cuda()
            log_probs = model.forward(x)
            minibatch_loss = criterion(log_probs, labels.long())
            loss = minibatch_loss
            loss.backward()
            for name, param in model.named_parameters():
                gradient[name] += param.grad.data
        return gradient

    def update_mask(self, masks, num_remove, gradient=None):
        for name in gradient:
            temp = torch.where(masks[name].cuda() == 0, torch.abs(gradient[name]),
                                -100000 * torch.ones_like(gradient[name]))
            sort_temp, idx = torch.sort(temp.view(-1), descending=True)
            masks[name].view(-1)[idx[:num_remove[name]]] = 1
        return masks
    
    # def init_mask(self,  gradient=None):
    #     for name in self.mask:
    #         num_init = torch.count_nonzero(self.mask[name])
    #         self.mask[name] = torch.zeros_like(self.mask[name])
    #         sort_temp, idx = torch.sort(torch.abs(gradient[name]).view(-1), descending=True)
    #         self.mask[name].view(-1)[idx[:num_init]] = 1
             

    def fire_mask(self, weights, masks, round):
        
        drop_ratio = 0.0001 / 2 * (1 + np.cos((round * np.pi) / (2000)))
    
        # logging.info(drop_ratio)
        num_remove = {}
        for name in masks:
                num_non_zeros = torch.sum(masks[name].cuda())
                num_remove[name] = math.ceil(drop_ratio * num_non_zeros)
     
        for name in masks:
            if num_remove[name]>0 and  "track" not in name and "running" not in name: 
                temp_weights = torch.where(masks[name].cuda() > 0, torch.abs(weights[name]),
                                        100000 * torch.ones_like(weights[name]))
                x, idx = torch.sort(temp_weights.view(-1).cuda())
                masks[name].view(-1)[idx[:num_remove[name]]] = 0
        return masks, num_remove



    def local_train(self, global_model, participant_id, round=None, temparature=10, alpha=0.3, global_mask= None, neurotoxin_mask =None, updates_dict =None):
        """ Do a local training over the received global model, return the update """
        initial_global_model_params = parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        global_model.cuda()
        for name, param in global_model.named_parameters():
            self.mask[name] =self.mask[name].cuda()
            param.data = param.data * self.mask[name]
        if self.num_remove!=None:
            gradient = self.screen_gradients(global_model)
            self.mask = self.update_mask(self.mask, self.num_remove, gradient)
        
        global_model.train()
        lr = self.helper.config.lr
        optimizer = torch.optim.SGD(global_model.parameters(), lr=lr,
            momentum=self.helper.config.momentum,
            weight_decay=self.helper.config.decay)
        for internal_epoch in range(self.helper.config.retrain_times):
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                output = global_model(inputs)
                loss = self.criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                for name, param in global_model.named_parameters():
                    param.grad.data = self.mask[name].cuda() * param.grad.data
                optimizer.step()

            self.mask, self.num_remove = self.fire_mask(global_model.state_dict(), self.mask, round) 
            

