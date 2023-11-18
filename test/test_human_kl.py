from optparse import Values
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import math
import argparse
import time
# from lib.utils import AverageMeter
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import random
import numpy as np
import sys
sys.path.append("../")
# import fp
sys.path.append("../")
from data.cifar10h.data_loader import input_dataset
from basic_models.resnet import *
from basic_models.vgg import *
from basic_models.lenet import *
from multi_train_utils.train_eval_utils import evaluate
from attacks.attack_utils import *
# from torchvision.datasets import CIFAR10, CIFAR100

# import foolbox
# from foolbox import PyTorchModel, accuracy, samples


batch_size =128
base_dir = "../DUBN_cifar/data/cifar/"
data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(32),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))]),
        "val": transforms.Compose([transforms.Resize(32),
                                   transforms.CenterCrop(32),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])}
class AverageMeter():
    def __init__(self):
        self.reset()
    def is_empty(self):
        return self.cnt == 0
    def reset(self):
        self.avg = 0.
        self.sum = 0.
        self.cnt = 0
    def append(self, val, n=1):
        self.sum += val*n
        self.cnt += n
        self.avg = self.sum / self.cnt


def worker_init_fn(worker_id):
	random.seed(worker_id+1)
def evaluate(model,test_loader):
    all_number = 0
    
    criterion = nn.KLDivLoss()
    # criterion = nn.CrossEntropyLoss()
    sum_kl=0
    cnt=0
    for idx,data in enumerate(test_loader):
        
        image,label,_,_ = data
        all_number += len(label)
        image,label = image.cuda(),label.cuda()
        pred = model(image)
        
        x = F.log_softmax(pred, dim=-1)
 
        # y = F.softmax(label, dim=-1)

        kl = criterion(x,label)
        sum_kl += kl.detach().cpu()
        cnt += 1
        # pred = torch.max(pred, dim=1)[1]
        # sum_num += torch.eq(pred, label).sum()

    # acc = (sum_num / all_number)*100
    return sum_kl/cnt
# def evaluate_pgd(model,test_loader):



def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    # cifar10 dataset
    

    
    # 实例化训练数据集
    trainset,testset,num_classes,num_training_samples = input_dataset()
    print('Using {} dataloader workers every process'.format(nw))
    test_loader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)
    print(batch_size)                                        
    # print('Using {} dataloader workers every process'.format(nw))
    if args.model =="resnet18":
        model = ResNet18().to(device)
    elif args.model == "resnet34":
        model = ResNet34().to(device)
    acc_total = []
    
    weight = args.weight

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
    
    acc = evaluate(model=model,test_loader=test_loader)
    print(weight)
    print(acc)
    # _,std_acc = evaluate_standard(test_loader, model)
    # print("dataset:{}, model:{}, noisetype:{}, noiseratio:{}, test_acc:{}, test_pgd_acc:{}".format(args.dataset,args.model,args.noise_type,str(args.noise_ratio),std_acc,pgd_acc))
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
    # parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lrf', type=float, default=0.0001)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--noise_type', default='clean')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--noise_ratio', type=float,default='0.1')
    parser.add_argument('--model', default='resnet34')
    parser.add_argument('--ifbest', default=False)
    parser.add_argument('--data_root_path', default="/data/zhaoxian/dataset/")
    # 数据集所在根目录
    # parser.add_argument('--weight', type=str, default='../weights/cifar10/clean/supcontrast/resne34/clean_sup_best.pth',help='initial weights path')
    # parser.add_argument('--weight', type=str, default='../weights/clean_ot_param/resnet34/best.pth',help='initial weights path')
    parser.add_argument('--weight', type=str, default='../weights/cifar10/aggre_label/supcontrast/resnet34/1_0.5_lr_splitby60/clean_sup_best.pth',help='initial weights path')
    # parser.add_argument('--weight', type=str, default='/data/zhaoxian/label_noise/train/weights/onlyright_40random/resnet18-best.pth',help='initial weights path')
    # parser.add_argument('--weight', type=str, default='/data/zhaoxian/label_noise/analysis/cifarn_workspace/weights/cifar10/worse_label/resnet18/only_wrong/best.pth',help='initial weights path')
    # parser.add_argument('--weight', type=str, default='/data/zhaoxian/label_noise/analysis/cifarn_workspace/weights/clean_ot_param/round1-199.pth',help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()

    main(opt)
