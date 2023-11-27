from optparse import Values
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import math
import argparse
import time
import torchvision
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
from basic_models.resnet import *
from basic_models.preact_resnet import *
from train.multi_train_utils.train_eval_utils import evaluate
from attacks.attack_utils import *
from torchvision.datasets import CIFAR10, CIFAR100
from loss import *
from foolbox import PyTorchModel, accuracy, samples
import torch.nn.functional as F
def Normalize(data):
    # m = np.mean(data)
    mx,_= data.max(1)
    mn,_= data.min(1)
    mx,mn = mx.unsqueeze(1),mn.unsqueeze(1)
    # print(mx.shape)
    # print(mn.shape)
    return (data-mn)/(mx-mn)
batch_size = 128
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

def cifar_c_testloader(corruption, data_dir, transforms,num_classes=10, 
    test_batch_size=200, num_workers=2):
    '''
    Returns:
        test_c_loader: corrupted testing set loader (original cifar10-C)
    CIFAR10-C has 50,000 test images. 
    The first 10,000 images in each .npy are of level 1 severity, and the last 10,000 are of level 5 severity.
    '''
    if num_classes==10:
        CIFAR = CIFAR10
        base_dir = os.path.join(data_dir, 'CIFAR10')
        base_c_path = os.path.join(data_dir, 'CIFAR10-C')
    elif num_classes==100:
        CIFAR = CIFAR100
        base_dir = os.path.join(data_dir, 'CIFAR100')
        base_c_path = os.path.join(data_dir, 'CIFAR-100-C')
    else:
        raise Exception('Wrong num_classes %d' % num_classes)

    #print('using in ', base_c_path)
    # test set:
    test_transform = transforms
    test_set = CIFAR("../DUBN_cifar/data/", train=False, transform=test_transform, download=False)

    test_set.data = np.load(os.path.join(base_c_path, '%s.npy' % corruption))
    test_set.targets = torch.LongTensor(np.load(os.path.join(base_c_path, 'labels.npy')))
    #print('loader for %s ready' % corruption)

    test_c_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return test_c_loader
def worker_init_fn(worker_id):
	random.seed(worker_id+1)
def evaluate(model,test_loader):
    all_number = 0
    for idx,data in enumerate(test_loader):
        
        image,label = data
        all_number += len(label)
        image,label = image.cuda(),label.cuda()
        pred = model(image)
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, label).sum()

    acc = (sum_num / all_number)*100
    return acc
# def evaluate_pgd(model,test_loader):

def evaluate_cifarc(model,args):
    CORRUPTIONS = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate',
        'jpeg_compression'
    ]
    test_seen_c_loader_list = []
    for corruption in CORRUPTIONS:
        test_c_loader = cifar_c_testloader(corruption=corruption, data_dir=args.data_root_path, transforms=data_transform['val'],num_classes=10, 
            test_batch_size=args.test_batch_size, num_workers=8)
        test_seen_c_loader_list.append(test_c_loader)
    test_c_losses, test_c_accs = [], []
    for corruption, test_c_loader in zip(CORRUPTIONS, test_seen_c_loader_list):
        test_c_batch_num = len(test_c_loader)
        print(test_c_batch_num) # each corruption has 10k * 5 images, each magnitude has 10k images
        ts = time.time()
        test_c_loss_meter, test_c_acc_meter = AverageMeter(), AverageMeter()
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(test_c_loader):
                images, targets = images.cuda(), targets.cuda()
                #images = normalize(images)
                logits = model(images)
                loss = F.cross_entropy(logits, targets)
                pred = logits.data.max(1)[1]
                acc = pred.eq(targets.data).float().mean()
                # append loss:
                test_c_loss_meter.append(loss.item())
                test_c_acc_meter.append(acc.item())

        print('%s test time: %.2fs' % (corruption, time.time()-ts))
        # test loss and acc of each type of corruptions:
        test_c_losses.append(test_c_loss_meter.avg)
        test_c_accs.append(test_c_acc_meter.avg)

        # print
        corruption_str = '%s: %.4f' % (corruption, test_c_accs[-1])
        print(corruption_str)
        # fp.write(corruption_str + '\n')
        # fp.flush()
    # mean over 16 types of attacks:
    test_c_loss = np.mean(test_c_losses)
    test_c_acc = np.mean(test_c_accs)

    # print
    avg_str = 'corruption acc: (mean) %.4f' % (test_c_acc)
    print(avg_str)
    # fp.write(avg_str + '\n')
    # fp.flush()

transform_aug = torchvision.transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
    ])
def evaluate_feature(model,test_loader,args):
    unif_all=0
    cnt = 0
    for data,label in test_loader:
        data = data.cuda()
        for j in range(10):
            indicate = torch.where(label==j)
            data_temp = data[indicate]
            if len(data_temp)>1:
                pred,fre=model(data_temp,if_feature=True)
                # print(fre.shape)
                # align = align_loss(fre[0],fre[1])
                fre = F.normalize(fre,dim=1)
                unif = uniform_loss(fre)

                unif_all += unif.detach().cpu()
                cnt +=1   
    unif_mean = unif_all/cnt
    # print(unif_mean)
    return unif_mean
def evaluate_feature_align(model,test_loader,args):
    unif_all=0
    cnt = 0
    align_all=0
    for data,label in test_loader:
        data = data.cuda()
       
        pred,fre=model(data,if_feature=True)
        # print(fre.shape)
        data_aug = transform_aug(data)
        # print(data_aug)
        pred,fre_aug=model(data_aug,if_feature=True)
        align = align_loss(fre,fre_aug)
        unif = uniform_loss(fre)
        align_all += align.detach().cpu()
        unif_all += unif.detach().cpu()
        cnt +=1   
    unif_mean = unif_all/cnt
    align_mean = align_all/cnt
    print(unif_mean)
    print(align_mean)
    return unif_mean
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.noise_type != "clean":
        weight_path = "../analysis/cifarn_workspace/weights/"+args.dataset+"/"+args.noise_type+"/"+args.model+"/"
        log_path = "./logs_"+args.dataset+"_"+args.noise_type+"_"+args.model+"_"+str(args.noise_ratio)
    else:
        weight_path = "../analysis/cifarn_workspace/weights/"+args.dataset+"/"+args.noise_type+"/"+args.model+"/"
        log_path = "./logs_"+args.dataset+"_"+args.noise_type+"_"+args.model
    
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    # 实例化训练数据集
    if args.dataset == "cifar10":
        test_data =  CIFAR10(base_dir, train=False, transform=data_transform["val"], download=False)
    if args.dataset =="cifar100":
        test_data =  CIFAR100(base_dir, train=False, transform=data_transform["val"], download=False)

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
    # model=PreActResNet18().to(device)
    weight = ""

    # 如果存在预训练权重则载入
    if weight != "":
        if os.path.exists(weight):
            weights_dict = torch.load(weight, map_location=device)
            new_state_dict = {}
            for k, v in weights_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
            # load_weights_dict = {k: v for k, v in weights_dict.items()
            #                     if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(state_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(weight))
    model.eval()
    print(weight)
    evaluate_feature_align(model,test_loader,args=args)
    std_acc,_ = evaluate_standard(test_loader, model)
    print(std_acc)
   
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
    parser.add_argument('--model', default='resnet18')
    parser.add_argument('--ifbest', default=False)
    parser.add_argument('--data_root_path', default="")
    # 数据集所在根目录
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()

    main(opt)
