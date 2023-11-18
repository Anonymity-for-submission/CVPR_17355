import sys

from attacks.attack_utils import evaluate_pgd,evaluate_standard
sys.path.append("../")
from data.cifar10h.cifar10h_dataset import CIFAR10H
from basic_models.resnet import *
from optparse import Values
import os
import math
import argparse

import torch
import torch.optim as optim
from torchvision import transforms
# import matplotlib.pyplot as plt
import random
import numpy as np

from torchvision.datasets import CIFAR10
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
batch_size =256
# base_dir = "../DUBN_cifar/data/cifar/"
base_dir = "../DUBN_cifar/cifar10/"
def worker_init_fn(worker_id):
	random.seed(worker_id+1)

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
   
    # weight = "../weights/cifar10/worse_label/supcontrast/resnet34/1_0.5_lr_splitby60/clean_sup_best.pth"
    # weight = "../weights/clean_ot_param/resnet34/best.pth"
    weight = "../weights/cifar10/clean/supcontrast/resnet34/clean_sup_last.pth"      
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

    # test_data =  CIFAR10H(root=base_dir, train=False, transform=data_transform["val"],type=args.type,if_original=args.original)
    test_data = CIFAR10(base_dir, train=False, transform=data_transform["val"], download=True)
    print('Using {} dataloader workers every process'.format(nw))
    test_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)
    # print('Using {} dataloader workers every process'.format(nw))
    if args.model =="resnet18":
        model = ResNet18().to(device)
    elif args.model == "resnet34":
        model = ResNet34().to(device)
    acc_total = []
    # 如果存在预训练权重则载入
    if weight != "":
        if os.path.exists(weight):
            weights_dict = torch.load(weight, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(weight))
    model.eval()
    

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf

        # validate
    # acc = evaluate_standard(test_loader = test_loader, model=model)
    loss,acc_pgd = evaluate_pgd(test_loader = test_loader,model=model,attack_iters=1,restarts=1)
    print(loss)
    print(acc_pgd)
    # print("dataset:{}, model:{}, type:{}, acc:{}, adversarial acc:{}".format(args.dataset,args.model,args.type,acc,acc_pgd))
    # acc_total.append(acc)
    # np.save("../results/test_acc_"+args.type+".npy",np.array(acc_total))
    #     # print(clipped_advs.shape)
    # values = [j*20 for j in range(11)]
    # plt.plot(values, acc_total, linewidth=4)
    # plt.title("Test Accuracy",fontsize=14)
    # plt.xlabel("epoch", fontsize=14)
    # plt.ylabel("accuracy", fontsize=14)
    # plt.savefig("../results/test_acc_"+args.type+".png")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lrf', type=float, default=0.0001)
    parser.add_argument('--type', default='all')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--noise_ratio', type=float,default='0.1')
    parser.add_argument('--model', default='resnet34')
    parser.add_argument('--ifbest', default=False)
    parser.add_argument('--original', default=True)
    # 数据集所在根目录

    parser.add_argument('--weights', type=str, default='../weight/resnet18-cifar-random-20.pth',help='initial weights path')
    
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()

    main(opt)