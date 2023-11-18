import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms
import sys
sys.path.append("../")
# from data.cifar10h.cifar10h_dataset import CIFAR10H
from basic_models.resnet import *
from torchvision.datasets import CIFAR10
import os
batch_size=256
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)
base_dir = "../DUBN_cifar/cifar10/"

# weight = "../weights/cifar10/worse_label/supcontrast/resnet34/1_0.5_lr_splitby60/clean_sup_last.pth"
weight = "../weights/clean_ot_param/resnet34/best.pth"
model="resnet34"
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

class Attack(object):
    def __init__(self, net, criterion):
        self.net = net
        self.criterion = criterion

    def fgsm(self, x, y, targeted=False, eps=0.03):
        x_adv = Variable(x.data, requires_grad=True)
        h_adv = self.net(x_adv)
        if targeted:
            cost = self.criterion(h_adv, y)
        else:
            cost = -self.criterion(h_adv, y)

        self.net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv.grad.sign_()
        x_adv = x_adv - eps*x_adv.grad
        x_adv = clamp(x_adv, lower_limit, upper_limit)


        h = self.net(x)
        h_adv = self.net(x_adv)

        return x_adv, h_adv, h
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
if model =="resnet18":
    model = ResNet18().to(device)
elif model == "resnet34":
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
atk=Attack(net=model,criterion=torch.nn.CrossEntropyLoss())
epsilon = (8 / 255.)-mu / std
alpha = (2 / 255.)-mu / std
pgd_loss = 0
pgd_acc = 0
n = 0
model.eval()
for i, (X, y) in enumerate(test_loader):
    X, y = X.cuda(), y.cuda()
    pgd_delta,_,_ = atk.fgsm(X, y)
    with torch.no_grad():
        output = model(pgd_delta)
        loss = F.cross_entropy(output, y)
        pgd_loss += loss.item() * y.size(0)
        pred = torch.max(output, dim=1)[1]
        pgd_acc += (pred == y).sum().item()
        n += y.size(0)
print(pgd_acc/n)