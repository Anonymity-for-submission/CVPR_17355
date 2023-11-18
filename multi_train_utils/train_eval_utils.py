from operator import le
import sys
import os
from tqdm import tqdm
import torch
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from .distributed_utils import reduce_value, is_main_process, warmup_lr_scheduler
sys.path.append("../")
from attacks.attack_utils import *

# import foolbox
# from foolbox import PyTorchModel, accuracy, samples
epsilon=[8/255.]
alpha = 2/255.

attack_iters = 7
def train_one_epoch_right(model, optimizer, data_loader, device, epoch, use_amp=False, warmup=True):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader)

    enable_amp = use_amp and "cuda" in device.type
    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)

    sample_num = 0
    for step, data in enumerate(data_loader):
        images,label,_,real_label,_= data
        # print(images.shape)
        sample_num += images.shape[0]
        # print(images.shape)
        with torch.cuda.amp.autocast(enabled=enable_amp):
            pred= model(images.to(device))
            # print(pred.shape)
            loss = loss_function(pred, label.long().to(device))

            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, label.to(device)).sum()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        loss = reduce_value(loss, average=True)
        accu_loss += loss.detach()

        # 在进程0中打印平均loss
        if is_main_process():
            info = "[epoch {}] loss: {:.3f}, train_acc: {:.3f}, lr: {:.5f}".format(
                epoch,
                accu_loss.item() / (step + 1),
                accu_num.item() / sample_num,
                optimizer.param_groups[0]["lr"])
            data_loader.desc = info

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        if lr_scheduler is not None:  # 如果使用warmup训练，逐渐调整学习率
            lr_scheduler.step()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return accu_loss.item() / (step + 1)
def train_one_epoch_adv(model, optimizer, data_loader, device, epoch, use_amp=False, warmup=True):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader)

    enable_amp = use_amp and "cuda" in device.type
    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)

    sample_num = 0
    for step, data in enumerate(data_loader):
        images,_,label= data
        # print(images.shape)
        sample_num += images.shape[0]
        
        # print(images.shape)
        with torch.cuda.amp.autocast(enabled=enable_amp):
            adv = attack_pgd(model, images.to(device), label.long().to(device), attack_iters=7, restarts=1, opt=None)
            pred= model(images.to(device)+adv)
            pred_clean = model(images.to(device))
            # print(pred.shape)
            loss = loss_function(pred, label.long().to(device))
            # loss += loss_function(pred_clean, label.long().to(device))
            pred_classes = torch.max(pred_clean, dim=1)[1]
            accu_num += torch.eq(pred_classes, label.to(device)).sum()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        loss = reduce_value(loss, average=True)
        accu_loss += loss.detach()

        # 在进程0中打印平均loss
        if is_main_process():
            info = "[epoch {}] loss: {:.3f}, train_acc: {:.3f}, lr: {:.5f}".format(
                epoch,
                accu_loss.item() / (step + 1),
                accu_num.item() / sample_num,
                optimizer.param_groups[0]["lr"])
            data_loader.desc = info

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        if lr_scheduler is not None:  # 如果使用warmup训练，逐渐调整学习率
            lr_scheduler.step()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return accu_loss.item() / (step + 1)
def train_one_epoch_adv_cifarn(model, optimizer, data_loader, device, epoch, use_amp=False, warmup=True):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader)

    enable_amp = use_amp and "cuda" in device.type
    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)

    sample_num = 0
    for step, data in enumerate(data_loader):
        images,label,_,_,_= data
        # print(images.shape)
        sample_num += images.shape[0]
        
        # print(images.shape)
        with torch.cuda.amp.autocast(enabled=enable_amp):
            adv = attack_pgd(model, images.to(device), label.to(device), attack_iters=7, restarts=1, opt=None)
            pred= model(images.to(device)+adv)
            pred_clean = model(images.to(device))
            # print(pred.shape)
            loss = loss_function(pred, label.long().to(device))
            # loss += loss_function(pred_clean, label.long().to(device))
            pred_classes = torch.max(pred_clean, dim=1)[1]
            accu_num += torch.eq(pred_classes, label.to(device)).sum()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        loss = reduce_value(loss, average=True)
        accu_loss += loss.detach()

        # 在进程0中打印平均loss
        if is_main_process():
            info = "[epoch {}] loss: {:.3f}, train_acc: {:.3f}, lr: {:.5f}".format(
                epoch,
                accu_loss.item() / (step + 1),
                accu_num.item() / sample_num,
                optimizer.param_groups[0]["lr"])
            data_loader.desc = info

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        if lr_scheduler is not None:  # 如果使用warmup训练，逐渐调整学习率
            lr_scheduler.step()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return accu_loss.item() / (step + 1)
def train_one_epoch_adv_cifarn_right(model, optimizer, data_loader, device, epoch, use_amp=False, warmup=True):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader)

    enable_amp = use_amp and "cuda" in device.type
    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)

    sample_num = 0
    for step, data in enumerate(data_loader):
        images,_,_,label,_= data
        # print(images.shape)
        sample_num += images.shape[0]
        
        # print(images.shape)
        with torch.cuda.amp.autocast(enabled=enable_amp):
            adv = attack_pgd(model, images.to(device), label.to(device), attack_iters=7, restarts=1, opt=None)
            pred= model(images.to(device)+adv)
            pred_clean = model(images.to(device))
            # print(pred.shape)
            loss = loss_function(pred, label.long().to(device))
            loss += loss_function(pred_clean, label.long().to(device))
            pred_classes = torch.max(pred_clean, dim=1)[1]
            accu_num += torch.eq(pred_classes, label.to(device)).sum()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        loss = reduce_value(loss, average=True)
        accu_loss += loss.detach()

        # 在进程0中打印平均loss
        if is_main_process():
            info = "[epoch {}] loss: {:.3f}, train_acc: {:.3f}, lr: {:.5f}".format(
                epoch,
                accu_loss.item() / (step + 1),
                accu_num.item() / sample_num,
                optimizer.param_groups[0]["lr"])
            data_loader.desc = info

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        if lr_scheduler is not None:  # 如果使用warmup训练，逐渐调整学习率
            lr_scheduler.step()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return accu_loss.item() / (step + 1)
def train_one_epoch_adv_clean(model, optimizer, data_loader, device, epoch, use_amp=False, warmup=False):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader)

    enable_amp = use_amp and "cuda" in device.type
    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)

    sample_num = 0
    for step, data in enumerate(data_loader):
        images,label= data
        # print(images.shape)
        sample_num += images.shape[0]
        
        # print(images.shape)
        with torch.cuda.amp.autocast(enabled=enable_amp):
            adv = attack_pgd(model, images.to(device), label.to(device), attack_iters=7, restarts=1, opt=None)
            pred= model(images.to(device)+adv)
            pred_clean = model(images.to(device))
            # print(pred.shape)
            loss = loss_function(pred, label.long().to(device))
            loss += loss_function(pred_clean, label.long().to(device))
            pred_classes = torch.max(pred_clean, dim=1)[1]
            accu_num += torch.eq(pred_classes, label.to(device)).sum()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        loss = reduce_value(loss, average=True)
        accu_loss += loss.detach()

        # 在进程0中打印平均loss
        if is_main_process():
            info = "[epoch {}] loss: {:.3f}, train_acc: {:.3f}, lr: {:.5f}".format(
                epoch,
                accu_loss.item() / (step + 1),
                accu_num.item() / sample_num,
                optimizer.param_groups[0]["lr"])
            data_loader.desc = info

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        if lr_scheduler is not None:  # 如果使用warmup训练，逐渐调整学习率
            lr_scheduler.step()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return accu_loss.item() / (step + 1)
def train_one_epoch(model, optimizer, data_loader, device, epoch, use_amp=False, warmup=True):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader)

    enable_amp = use_amp and "cuda" in device.type
    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)

    sample_num = 0
    for step, data in enumerate(data_loader):
        images,label= data
        # print(images.shape)
        sample_num += images.shape[0]
        # print(images.shape)
        with torch.cuda.amp.autocast(enabled=enable_amp):
            pred= model(images.to(device))
            # print(pred.shape)
            loss = loss_function(pred, label.long().to(device))

            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, label.to(device)).sum()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        loss = reduce_value(loss, average=True)
        accu_loss += loss.detach()

        # 在进程0中打印平均loss
        if is_main_process():
            info = "[epoch {}] loss: {:.3f}, train_acc: {:.3f}, lr: {:.5f}".format(
                epoch,
                accu_loss.item() / (step + 1),
                accu_num.item() / sample_num,
                optimizer.param_groups[0]["lr"])
            data_loader.desc = info

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        if lr_scheduler is not None:  # 如果使用warmup训练，逐渐调整学习率
            lr_scheduler.step()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return accu_loss.item() / (step + 1)
def train_one_epoch_onlyright(model, optimizer, data_loader, device, epoch, use_amp=False, warmup=True):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader)

    enable_amp = use_amp and "cuda" in device.type
    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)

    sample_num = 0
    for step, data in enumerate(data_loader):
        images,label,real_label= data
        # print(images.shape)
        sample_num += images.shape[0]
        # print(images.shape)
        with torch.cuda.amp.autocast(enabled=enable_amp):
            pred= model(images.to(device))
            # print(pred.shape)
            loss = loss_function(pred, real_label.long().to(device))

            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, real_label.to(device)).sum()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        loss = reduce_value(loss, average=True)
        accu_loss += loss.detach()

        # 在进程0中打印平均loss
        if is_main_process():
            info = "[epoch {}] loss: {:.3f}, train_acc: {:.3f}, lr: {:.5f}".format(
                epoch,
                accu_loss.item() / (step + 1),
                accu_num.item() / sample_num,
                optimizer.param_groups[0]["lr"])
            data_loader.desc = info

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        if lr_scheduler is not None:  # 如果使用warmup训练，逐渐调整学习率
            lr_scheduler.step()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return accu_loss.item() / (step + 1)
def train_one_epoch_part(model, optimizer, rate,data_loader, device, epoch, use_amp=False, warmup=True):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader)

    enable_amp = use_amp and "cuda" in device.type
    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)

    sample_num = 0
    for step, data in enumerate(data_loader):
        images,label,_,_= data
        images = images[:int(len(images)*rate)]
        label = label[:int(len(label)*rate)]
        # print(images.shape)
        # print(label.shape)
        # print(images.shape)
        sample_num += images.shape[0]
        # print(images.shape)
        with torch.cuda.amp.autocast(enabled=enable_amp):
            pred= model(images.to(device))
            # print(pred.shape)
            loss = loss_function(pred, label.long().to(device))

            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, label.to(device)).sum()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        loss = reduce_value(loss, average=True)
        accu_loss += loss.detach()

        # 在进程0中打印平均loss
        if is_main_process():
            info = "[epoch {}] loss: {:.3f}, train_acc: {:.3f}, lr: {:.5f}".format(
                epoch,
                accu_loss.item() / (step + 1),
                accu_num.item() / sample_num,
                optimizer.param_groups[0]["lr"])
            data_loader.desc = info

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        if lr_scheduler is not None:  # 如果使用warmup训练，逐渐调整学习率
            lr_scheduler.step()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return accu_loss.item() / (step + 1)

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    # 验证集样本个数
    num_samples = len(data_loader.dataset)
    pred_list = []
    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)
    loss_all = 0
    loss_function = torch.nn.CrossEntropyLoss()
    # 在进程0中打印验证进度
    if is_main_process():
        data_loader = tqdm(data_loader)
    cnt = 0
    for step, data in enumerate(data_loader):
        images,labels= data
        pred = model(images.to(device))
        loss_all += loss_function(pred,labels.to(device))
        # pred_list.extend(pred.detach().cpu())
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.long().to(device)).sum()
        cnt += 1
    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    sum_num = reduce_value(sum_num, average=False)
    acc = sum_num.item() / num_samples
    loss_all = loss_all/cnt
    return acc,loss_all

def evaluate_confidence(model, data_loader, device):
    model.eval()

    # 验证集样本个数
    num_samples = 0

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)
    # temp = 0
    image_symbol = []
    # 在进程0中打印验证进度
    if is_main_process():
        data_loader = tqdm(data_loader)

    for step, data in enumerate(data_loader):
        images,label,index,real_label = data

        if(images.shape[0] !=0):
            # temp += 1
            # print(images.shape)
            pred_final = torch.zeros(10)
            num_samples += images.shape[0]

            pred= model(images.to(device))
            pred = torch.nn.functional.softmax(pred, 1)
            # print(len(pred))
            for j in range(len(pred)):
                pred_final[j] = pred[j][label[j]]
            # pred = pred[label]
            # sum_num += torch.eq(pred, real_label.to(device)).sum()

    # 等待所有进程计算完毕
    # if device != torch.device("cpu"):
    #     torch.cuda.synchronize(device)

    # sum_num = reduce_value(sum_num, average=False)
    # acc = sum_num.item() / num_samples

    return pred

def evaluate_train(model, data_loader, device,if_original=False):
    model.eval()

    # 验证集样本个数
    num_samples = len(data_loader.dataset)

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)

    # 在进程0中打印验证进度
    if is_main_process():
        data_loader = tqdm(data_loader)

    for step, data in enumerate(data_loader):
        images,labels,_,_,_= data
        pred = model(images.to(device))
    
        pred = torch.max(pred, dim=1)[1]
        if not if_original:
            labels = torch.max(labels,dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    sum_num = reduce_value(sum_num, average=False)
    acc = sum_num.item() / num_samples

    return acc