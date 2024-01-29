import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import sys
sys.path.append("../")
from data.cifar10h.cifar10h_dataset import CIFAR10H
from basic_models.resnet import *
import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import warnings
os.environ["CUDA_VISIBLE_DEVICES"]="5"
warnings.filterwarnings("ignore")
batch_size = 256
base_dir = "../DUBN_cifar/cifar10/"
model = "resnet34"
weight = "../weights/clean_ot_param/resnet34/best.pth"
# %matplotlib inline
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def cw_l2_attack(model, images, labels, targeted=False, c=1e-4, kappa=0, max_iter=1000, learning_rate=0.01) :

    images = images.to(device)     
    labels = labels.to(device)

    # Define f-function
    def f(x) :

        outputs = model(x)
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.byte())
        
        # If targeted, optimize for making the other class most likely 
        if targeted :
            return torch.clamp(i-j, min=-kappa)
        
        # If untargeted, optimize for making the other class most likely 
        else :
            return torch.clamp(j-i, min=-kappa)
    
    w = torch.zeros_like(images, requires_grad=True).to(device)

    optimizer = optim.Adam([w], lr=learning_rate)

    prev = 1e10
    
    for step in range(max_iter) :

        a = 1/2*(nn.Tanh()(w) + 1)

        loss1 = nn.MSELoss(reduction='sum')(a, images)
        loss2 = torch.sum(c*f(a))

        cost = loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Early Stop when loss does not converge.
        if step % (max_iter//10) == 0 :
            if cost > prev :
                print('Attack Stopped due to CONVERGENCE....')
                return a
            prev = cost
        
        print('- Learning Progress : %2.2f %%        ' %((step+1)/max_iter*100), end='\r')

    attack_images = clamp(1/2*(nn.Tanh()(w) + 1),lower_limit,upper_limit)

    return attack_images
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

    # weight = "../weights/cifar10/clean/supcontrast/resnet34/clean_sup_best.pth"      
# weight_path = "../../analysis/cifarn_workspace/weights/clean_ot_param/"
    # weight_path = "../weights/cifar10/worse_label/supcontrast/resnet18/0.5_0.2_lr_splitby20/"    
    # weight = weight_path+"clean_sup_best.pth"
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

# cifar10 dataset
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(32),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))]),
    "val": transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])}

test_data =  CIFAR10H(root=base_dir, train=False, transform=data_transform["val"],type="all",if_original=True)
print('Using {} dataloader workers every process'.format(nw))
test_loader = torch.utils.data.DataLoader(test_data,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=nw)
# print('Using {} dataloader workers every process'.format(nw))
if model =="resnet18":
    model = ResNet18().to(device)
elif model == "resnet34":
    model = ResNet34().to(device)
acc_total = []
# 
if weight != "":
    if os.path.exists(weight):
        weights_dict = torch.load(weight, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                            if model.state_dict()[k].numel() == v.numel()}
        print(model.load_state_dict(load_weights_dict, strict=False))
    else:
        raise FileNotFoundError("not found weights file: {}".format(weight))
model.eval()

correct = 0
total = 0
total_loss= 0 
for images, labels in test_loader:
    
    images = cw_l2_attack(model, images, labels, targeted=False, c=0.1)
    labels = labels.to(device)
    outputs = model(images)
    total_loss += F.cross_entropy(outputs, labels).detach().cpu()
    _, pre = torch.max(outputs.data, 1)
       
    total += 1
    correct += (pre == labels).sum().detach().cpu()
    
    # imshow(torchvision.utils.make_grid(images.cpu().data, normalize=True), [normal_data.classes[i] for i in pre])
print(weight)
print('Accuracy of test text: %f %%' % (100 * float(correct) / total))
print('Loss: %f %%' % (100 * float(total_loss) / total))
