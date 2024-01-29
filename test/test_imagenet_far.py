import sys
import os
import torch
import torchvision
sys.path.append("../")
from data.imagenet_data import trainset
from basic_models.resnet import *

os.environ["CUDA_VISIBLE_DEVICES"]="4"
traindata=trainset()
# print(traindata.data.shape)
# print(traindata.label.shape)
batch_size = 128
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
train_loader = torch.utils.data.DataLoader(traindata,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=nw)
model = ResNet34().cuda()

# weight = "../weights/cifar10/clean/supcontrast/resnet34/clean_sup_best.pth"
weight="../weights/clean_ot_param/resnet34/best.pth"
# 
if weight != "":
    if os.path.exists(weight):
        weights_dict = torch.load(weight)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                            if model.state_dict()[k].numel() == v.numel()}
        print(model.load_state_dict(load_weights_dict, strict=False))
    else:
        raise FileNotFoundError("not found weights file: {}".format(weight))

model.eval()
sum_num = torch.zeros(1).cuda()
sum_loss = torch.zeros(1).cuda()
loss_function = torch.nn.CrossEntropyLoss()
cnt = 0
for data in train_loader:
    image,label = data
    pred = model(image.cuda())
    sum_loss += loss_function(pred, label.cuda()).detach().cpu()
    pred = torch.max(pred, dim=1)[1]
    sum_num += torch.eq(pred, label.long().cuda()).sum().detach().cpu()

    cnt += 1
print(sum_num/cnt)
print(sum_loss/cnt)
