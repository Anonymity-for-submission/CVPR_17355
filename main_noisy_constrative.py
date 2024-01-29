from lib2to3.pgen2.literals import evalString
from optparse import Values
import os
import math
import argparse
from tqdm import tqdm
import torch
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import random
import numpy as np
import sys
from loss import SupConLoss
# sys.path.append('../../')
# sys.path.append("../../")
from data.datasets import input_dataset
# from multi_train_utils import train_one_epoch
from basic_models.resnet import *
from basic_models.vgg import *
from basic_models.lenet import *
from multi_train_utils.distributed_utils import reduce_value, is_main_process, warmup_lr_scheduler
from multi_train_utils.train_eval_utils import evaluate_train,train_one_epoch,train_one_epoch_right

temperature=0.1
criterion = SupConLoss(temperature)

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
batch_size =400
base_dir = "../../DUBN_cifar/data/cifar/"

def worker_init_fn(worker_id):
	random.seed(worker_id+1)
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    weight_path = "../weights/"+args.dataset+"/"+args.noise_type+"/supcontrast/"+args.model+"/1_0.5_lr_splitby60/new/"
    
    if os.path.exists(weight_path) is False:
        os.makedirs(weight_path)
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    
    noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
    train_dataset_clean,test_dataset_clean,num_classes,num_training_samples_clean = input_dataset(args.dataset,"../../DUBN_cifar/data/",args.noise_type,  "../../data/CIFAR-10_human.pt",True,only_right=True)
    train_dataset,_,_,num_training_samples= input_dataset(args.dataset, "../../DUBN_cifar/data/",args.noise_type, "../../data/CIFAR-10_human.pt",True,only_wrong=True)
    print('Using {} dataloader workers every process'.format(nw))
    dataset_concate = torch.utils.data.ConcatDataset([train_dataset_clean,train_dataset])
    train_loader = torch.utils.data.DataLoader(dataset_concate,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=nw)
    test_loader_clean = torch.utils.data.DataLoader(test_dataset_clean,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)

    # print('Using {} dataloader workers every process'.format(nw))
    if args.model =="resnet18":
        model = ResNet18().to(device)
    elif args.model == "resnet34":
        model = ResNet34(class_no=10).to(device)
    elif args.model == "resnet50":
        model = ResNet50().to(device)
    elif args.model == "vgg16":
        model = vgg16().to(device)
    elif args.model == "lenet":
        model = Lenet5(3).to(device)
    # acc_total = []
   
    weight = "../weights/cifar10/worse_label/supcontrast/resnet34/1_0.5_lr_splitby60/newclean_sup_best.pth"

# # 
    if weight != "":
        if os.path.exists(weight):
            weights_dict = torch.load(weight, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(weight))
    # model.eval()

    
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-4)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    loss_all=[]
    loss_ce_all=[]
    loss_loss1_all=[]
    loss_loss2_all=[]
    loss_mse_all=[]
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60,120,180], gamma=0.1)
    best_acc=0
    for epoch in range(args.epochs):
        # train
        model.train()
        loss_function = torch.nn.CrossEntropyLoss()
        # loss_mse = torch.nn.MSELoss()
        accu_loss = torch.zeros(1).to(device)  #
        accu_num = torch.zeros(1).to(device) 
          # 
        optimizer.zero_grad()

        lr_scheduler_local = None
        if epoch == 0 and args.warmup is True:  # 
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(train_loader) - 1)

            lr_scheduler_local = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

        # 
        # if is_main_process():
        data_loader = tqdm(train_loader)

        enable_amp = args.use_amp and "cuda" in device.type
        scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)

        sample_num = 0
        for step, data in enumerate(data_loader):
            images,label,_,real_label,con= data
            number = len(images)
        
            # print(images.shape)
            # print(con.shape)
            # print(real_label)
            # print(con)
            clean_indicate = torch.where(con==0)
            # print(clean_indicate[0].shape)
            clean_data = images[clean_indicate[0]]
            clean_label = label[clean_indicate[0]]
            clean_real_label = real_label[clean_indicate[0]]

            noisy_indicate = torch.where(con==1)
            # print(noisy_indicate[0].shape)
            noisy_data = images[noisy_indicate[0]]
            noisy_label = label[noisy_indicate[0]]
            noisy_real_label = real_label[noisy_indicate[0]]

            noisy_label_onhot = torch.zeros([len(noisy_data),10])
            for i in range(len(noisy_label_onhot)):
                noisy_label_onhot[i][real_label[i]] = 0.5
                noisy_label_onhot[i][noisy_label[i]] = 0.5
            
            pred_clean,features_clean = model(clean_data.to(device),if_feature=True)
            pred_noisy,featuren_noisy = model(noisy_data.to(device),if_feature=True)
            pred_noisy = torch.softmax(pred_noisy,dim=1)
            # print(pred_noisy.shape)
            # print(noisy_label_onhot.shape)
            # loss_mse_num = loss_mse(noisy_label_onhot.to(device),pred_noisy.to(device))
            # loss_ruan = loss_mse(pred_noisy)
            sample_num += images.shape[0]
            # # print(images.shape)
            bsz_clean = clean_label.shape[0]
            bsz_noisy = noisy_label.shape[0]
            
            bsz = min(bsz_clean,bsz_noisy)
            images = torch.cat([images, images], dim=0)
            # label_temp = torch.cat([clean_label[:bsz],noisy_label[:bsz]], dim=0)
            real_label_temp = torch.cat([real_label,real_label], dim=0)

            pred,features = model(images.to(device),if_feature=True)
            f1, f2 = torch.split(features, [number, number], dim=0)
            p1, p2 = torch.split(pred, [number, number], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            # # pred,fre = model(images.to(device),if_feature=True)
            # print(features.shape)
            with torch.cuda.amp.autocast(enabled=enable_amp):
            #     pred= model(images.to(device))
            #     # print(pred.shape)
            # 
                # print(real_label_temp)
                loss_ce = loss_function(pred_clean, clean_real_label.long().to(device))

                loss1 = criterion(features, real_label = real_label)
            #     # loss2 = criterion(features, labels = label)
            # # 
                loss2 = criterion(features, labels=label, real_label = real_label)
                loss = loss_ce+0.2*loss1+0.1*loss2
                
                pred_classes = torch.max(pred_clean, dim=1)[1]
                accu_num += torch.eq(pred_classes, clean_real_label.to(device)).sum()
        
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            loss = reduce_value(loss, average=True)
            accu_loss += loss.detach().cpu()
            # # 在进程0中打印平均loss
            # if is_main_process():
            info = "[epoch {}] loss: {:.3f}, train_acc: {:.3f}, lr: {:.5f}".format(
                epoch,
                accu_loss.item() / (step + 1),
                accu_num.item() / sample_num,
                optimizer.param_groups[0]["lr"])
            data_loader.desc = info

            # if not torch.isfinite(loss):
            #     print('WARNING: non-finite loss, ending training ', loss)
            #     sys.exit(1)

            if lr_scheduler_local is not None:  # 如果使用warmup训练，逐渐调整学习率
                lr_scheduler_local.step()
        loss_all.append(np.array(loss.mean().detach().cpu()))
        loss_ce_all.append(np.array(loss_ce.mean().detach().cpu()))
        loss_loss1_all.append(np.array(loss1.mean().detach().cpu()))
        loss_loss2_all.append(np.array(loss2.mean().detach().cpu()))
        # loss_mse_all.append(np.array(loss_mse_num.mean().detach().cpu()))
        # 
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        scheduler.step()

        # validate
        acc_clean = evaluate_train(model=model,
                       data_loader=test_loader_clean,
                       device=device,if_original=True)
        # tags = ["loss", "accuracy_clean","learning_rate"]
        # tb_writer.add_scalar(tags[0], mean_loss, epoch)
        # tb_writer.add_scalar(tags[1], acc_clean, epoch)
        # # tb_writer.add_scalar(tags[7], acc, epoch)
        # tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)


        print("[epoch {}] accuracy_clean: {}".format(epoch, round(acc_clean, 3)))
        # if best_acc<acc_clean:
        #     best_acc = acc_clean
        #     torch.save(model.state_dict(),weight_path+"clean_sup_best.pth")
        # torch.save(model.state_dict(),weight_path+"clean_sup_last.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--lrf', type=float, default=0.0001)
    parser.add_argument('--noise_type', default='worse_label')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--noise_ratio', type=float,default='0.1')
    parser.add_argument('--model', default='resnet34')
    parser.add_argument('--ifbest', default=False)
    # 

    parser.add_argument('--weights', type=str, default='../weight/resnet18-cifar-random-20.pth',
                        help='initial weights path')
    
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--warmup', type=bool, default=True)
    parser.add_argument('--use_amp', type=bool, default=False)
    opt = parser.parse_args()

    main(opt)
