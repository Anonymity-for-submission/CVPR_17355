from cam_derivatives import GradCAM
import sys
import os
# from utils import visualize_cam, Normalize
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from tqdm import tqdm
sys.path.append("../")
from data.datasets import input_dataset
# sys.path.append("../../")
# from sy_data.dataset import NoiseDataset
sys.path.append("../../../")
from basic_models.resnet import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18().cuda()

batch_size=256
resnet_model_dict = dict(type='resnet', arch=model, layer_name='layer4', input_size=(32, 32))
gradcam = GradCAM(resnet_model_dict, True)

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
train_dataset,test_dataset,num_classes,num_training_samples = input_dataset("cifar10","/data/zhaoxian/label_noise/DUBN_cifar/cifar10/","aggre_label", "../data/CIFAR-10_human.pt",True,only_wrong=True )
train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                            num_workers=nw)

all_noisy_clean = np.zeros(200)
all_noisy_ot = np.zeros(200)
# all_ot = np.zeros(200)
for i in range(200):
    weight = "../weights/clean/round_"+str(i)+".pth"
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
    noisy_clean = 0
    noisy_ot = 0
    for image,label,_,real_label,_ in tqdm(train_loader):
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
        
        temp = image[0].unsqueeze(0)
        mask, _ = gradcam(temp.cuda(),class_idx=label[0])
        # print(mask.shape)
        mask_r, _ = gradcam(temp.cuda(),class_idx=real_label[0])

        ot = (label[0]+1)%10
        if(ot == real_label[0]):
                ot = (ot + 1)%10

        mask_o, _ = gradcam(temp.cuda(),class_idx=ot)
        # all_noisy_clean[i]
        mask = mask.sum()
        mask_r = mask_r.sum()
        mask_o=mask_o.sum()
        if mask_r > 1e-5:
            noisy_clean += mask/mask_r
        elif mask > 1e-5:
            noisy_clean += 10
        else:
            noisy_clean += 1
        if mask_o > 1e-5:
             
            noisy_ot += mask/mask_o
        elif mask > 1e-5:
            noisy_ot += 10
        else:
            noisy_ot += 1
        # all_noisy += mask.sum()
        # all_clean += mask_r.sum()
        # all_ot += mask_o.sum()
        cnt += 1
    all_noisy_clean[i] = noisy_clean/cnt
    all_noisy_ot[i] = noisy_ot/cnt
    print(all_noisy_clean[i],all_noisy_ot[i])
np.save("./results/noisy_clean_aggre.npy",all_noisy_clean)
np.save("./results/noisy_ot_aggre.npy",all_noisy_ot)
    # print(all_noisy/cnt)
    # print(all_clean/cnt)
    # print(all_ot/cnt)
        # print(heatmap)
        # print(result)
        # mask_pp, _ = gradcam_pp(temp.cuda())
        # print(mask_pp.shape)
    # images = make_grid(torch.cat(images, 0), nrow=5)
    # print(images.shape)
    # output_dir = 'outputs_cifar100_our'
    # os.makedirs(output_dir, exist_ok=True)
    # output_name = "label_"+str(label_temp[0].detach().cpu())+"_"+str(cnt)+'.png'
    # output_path = os.path.join(output_dir, output_name)
    # cnt += 1
    
    # save_image(images, output_path)