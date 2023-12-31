from CAM import GradCAM, GradCAMpp
import sys
import os
from utils import visualize_cam, Normalize
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
sys.path.append("../")
from data.datasets import input_dataset
# sys.path.append("../../")
# from sy_data.dataset import NoiseDataset
sys.path.append("../../../")
from basic_models.resnet import *
mean = (0.4914, 0.4822, 0.4465)
std=(0.2471, 0.2435, 0.2616)
def showimg(image,path):
    fig, AX = plt.subplots(1,4, figsize=(105, 42))
    print(np.max(image[3]),np.min(image[3]))
    AX[0].imshow(image[0])
    AX[1].imshow(image[1])
    AX[2].imshow(image[2])
    AX[3].imshow(image[3])
    # AX[4].imshow(image[4])
    # plt.show()
    plt.savefig(path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18().cuda()
weight = "../weights/clean_ot_param/round1-199.pth"
batch_size=512
resnet_model_dict = dict(type='resnet', arch=model, layer_name='layer4', input_size=(32, 32))
resnet_gradcam = GradCAM(resnet_model_dict, True)
resnet_gradcampp = GradCAMpp(resnet_model_dict, True)
cam_dict = dict()
cam_dict['resnet'] = [resnet_gradcam, resnet_gradcampp]
cnt =0
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
data_transform = {
        "train": transforms.Compose([
            # transforms.RandomResizedCrop(32),
            #                          transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))]),
        "val": transforms.Compose([transforms.Resize(32),
                                   transforms.CenterCrop(32),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])} 

# train_dataset = NoiseDataset("cifar10","random",0.09,data_transform["train"],only_wrong=True)
train_dataset,test_dataset,num_classes,num_training_samples = input_dataset("cifar10","../DUBN_cifar/cifar10/","worse_label", "../data/CIFAR-10_human.pt",True,only_wrong=True )
train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=nw)
if weight != "":
        if os.path.exists(weight):
            weights_dict = torch.load(weight, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(weight))
model.eval()
cnt=0
for image,label,_,real_label,_ in train_loader:
    # pred = model(image.to(device))
    # indicates=[]
    # maxk = max((1,k))
    # y_resize = labels.view(-1,1)
    # pred = torch.max(pred, dim=1)[1]
    # _,pred = pred.topk(maxk,1,True,True)
    # indicates = torch.where(pred==label.to(device))
    # img_temp = image[indicates]
    # label_temp = label[indicates]
    # real_label=real_label[indicates]
    # sum_num += torch.eq()
    images =[]
    for gradcam, gradcam_pp in cam_dict.values():
        temp = image[0].unsqueeze(0)
        mask, _ = gradcam(temp.cuda(),class_idx=label[0])
        # print(mask.shape)
        temp  = temp*0.24 +0.48
        heatmap, result = visualize_cam(mask.cpu(), temp.cpu())

        mask_r, _ = gradcam(temp.cuda(),class_idx=real_label[0])

        ot = (label[0]+1)%10
        if(ot == real_label[0]):
             ot = (ot + 1)%10

        mask_o, _ = gradcam(temp.cuda(),class_idx=ot)
        # print(heatmap)
        # print(result)
        # mask_pp, _ = gradcam_pp(temp.cuda())
        # print(mask_pp.shape)
        heatmap_pp, result_pp = visualize_cam(mask_r.cpu(), temp.cpu())
        heatmap_o, result_pp = visualize_cam(mask_o.cpu(), temp.cpu())
        
        images.append(torch.stack([temp.squeeze().cpu(), heatmap, heatmap_pp, heatmap_o], 0))
        
    
    result = images[0].permute(0,2,3,1)
    
    result = np.array(result)
    # # result = result*std+mean
    output_dir = 'test_worse'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_name = "worse_label_"+str(label[0].detach().cpu())+"_"+"real_"+str(real_label[0])+"ot_"+str(ot)+str(cnt)+'.png'
    output_path = os.path.join(output_dir, output_name)
    showimg(result,output_path)
    # print(images[0].shape)
    cnt += 1
    # images = make_grid(torch.cat(images, 0), nrow=5)
    # print(images.shape)
    # output_dir = 'outputs_cifar100_our'
    # os.makedirs(output_dir, exist_ok=True)
    # output_name = "label_"+str(label_temp[0].detach().cpu())+"_"+str(cnt)+'.png'
    # output_path = os.path.join(output_dir, output_name)
    # cnt += 1
    
    # save_image(images, output_path)
