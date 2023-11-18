import io
import json
import os
import pickle
import torch
import torch.nn as nn
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import numpy as np
import pandas as pd
import scipy.stats
import pathlib
import PIL.Image
# import cifar10
import sys
sys.path.append("../")
from basic_models.resnet import *
from basic_models.vgg import *
from basic_models.lenet import *
import argparse
from torchvision.transforms import transforms
from tensorflow import Variable
from PIL import Image

cifar10_label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def parse_option():
    parser = argparse.ArgumentParser('STL-10 Representation Learning with Alignment and Uniformity Losses')
    parser.add_argument('--encoder_checkpoint', type=str, default='./results/align1alpha2_unif1t2/encoder-2.pth', help='Encoder checkpoint to evaluate')
    parser.add_argument('--align_w', type=float, default=1, help='Alignment loss weight')
    parser.add_argument('--unif_w', type=float, default=1, help='Uniformity loss weight')
    parser.add_argument('--align_alpha', type=float, default=2, help='alpha in alignment loss')
    parser.add_argument('--unif_t', type=float, default=2, help='t in uniformity loss')
    parser.add_argument('--layer_index', type=int, default=-1, help='Evaluation layer')
    parser.add_argument('--batch_size', type=int, default=768, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate. Default is linear scaling 0.12 per 256 batch size')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='Learning rate decay rate')
    parser.add_argument('--lr_decay_epochs', default=[155, 170, 185], nargs='*', type=int,
                        help='When to decay learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 weight decay')
    parser.add_argument('--feat_dim', type=int, default=2, help='Feature dimensionality')

    parser.add_argument('--num_workers', type=int, default=20, help='Number of data loader workers to use')
    parser.add_argument('--log_interval', type=int, default=40, help='Number of iterations between logs')
    parser.add_argument('--gpu', type=int, default='0', help='One GPU to use')

    parser.add_argument('--data_folder', type=str, default='./data', help='Path to data')
    parser.add_argument('--result_folder', type=str, default='./results', help='Base directory to save model')

    opt = parser.parse_args()

    if opt.lr is None:
        opt.lr = 0.12 * (opt.batch_size / 256)

    opt.gpu = torch.device('cuda', opt.gpu)

    opt.save_folder = os.path.join(
        opt.result_folder,
        f"align{opt.align_w:g}alpha{opt.align_alpha:g}_unif{opt.unif_w:g}t{opt.unif_t:g}"
    )
    os.makedirs(opt.save_folder, exist_ok=True)

    return opt

def load_new_test_data(version_string='', load_tinyimage_indices=False, transforms=transforms.Compose([transforms.Resize(32),
                                   transforms.CenterCrop(32),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])
):
    data_path = os.path.join(os.path.dirname(__file__), '../data/zero-shot/data/cifar10-1/')
    filename = 'cifar10.1'
    if version_string == '':
        version_string = 'v7'
    if version_string in ['v4', 'v6', 'v7']:
        filename += '_' + version_string
    else:
        raise ValueError('Unknown dataset version "{}".'.format(version_string))
    label_filename = filename + '_labels.npy'
    imagedata_filename = filename + '_data.npy'
    
    label_filepath = os.path.abspath(os.path.join(data_path, label_filename))
    # print(label_filepath)
    imagedata_filepath = os.path.abspath(os.path.join(data_path, imagedata_filename))
    print('Loading labels from file {}'.format(label_filepath))
    assert pathlib.Path(label_filepath).is_file()
    labels = np.load(label_filepath)
    print('Loading image data from file {}'.format(imagedata_filepath))
    assert pathlib.Path(imagedata_filepath).is_file()
    imagedata = np.load(imagedata_filepath)
    print(imagedata.shape)
    assert len(labels.shape) == 1
    assert len(imagedata.shape) == 4
    assert labels.shape[0] == imagedata.shape[0]
    assert imagedata.shape[1] == 32
    assert imagedata.shape[2] == 32
    assert imagedata.shape[3] == 3
    if version_string == 'v6' or version_string == 'v7':
        assert labels.shape[0] == 2000
    elif version_string == 'v4':
        assert labels.shape[0] == 2021

    transformed_images = []
    if transforms is not None:
        for image in imagedata:
            img = Image.fromarray(np.uint8(image)).convert("RGB")
            transformed_image = transforms(img)
            # print(transformed_image)
            transformed_images.append(transformed_image)
    print(len(transformed_images))

    labels = torch.from_numpy(labels)
    if not load_tinyimage_indices:
        return transformed_images, labels
    else:
        ti_indices_data_path = os.path.join(os.path.dirname(__file__), '../other_data/')
        ti_indices_filename = 'cifar10.1_' + version_string + '_ti_indices.json'
        ti_indices_filepath = os.path.abspath(os.path.join(ti_indices_data_path, ti_indices_filename))
        print('Loading Tiny Image indices from file {}'.format(ti_indices_filepath))
        assert pathlib.Path(ti_indices_filepath).is_file()
        with open(ti_indices_filepath, 'r') as f:
            tinyimage_indices = json.load(f)
        assert type(tinyimage_indices) is list
        assert len(tinyimage_indices) == labels.shape[0]
        return imagedata, labels, tinyimage_indices

def main():
    opt = parse_option()

    torch.cuda.set_device(opt.gpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 读数据
    test_dataset = load_new_test_data(version_string='v4')
#     transforms=transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
# ])
    print(len(test_dataset))
    criterion = nn.CrossEntropyLoss()
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    # print(test_loader[0].shape)
    # print(len(test_loader))
    encoder = ResNet34()
    # encoder.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    # encoder.fc = torch.nn.Linear(2, n_class)
    encoder = encoder.to(device)
    # weight = "../weights/only_right/worse/round_199.pth"
    # weight = "../weights/clean_ot_param/resnet34/best.pth"
    # weight = "../weights/cifar10/clean/supcontrast/resnet34/clean_sup_best.pth"
    weight = "../weights/cifar10/worse_label/supcontrast/resnet34/1_0.5_lr_splitby60/new/clean_sup_best.pth"
    if weight != "":
        if os.path.exists(weight):
            weights_dict = torch.load(weight, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                if encoder.state_dict()[k].numel() == v.numel()}
            print(encoder.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(weight))
    encoder.eval()

    # encoder.load_state_dict(torch.load(, map_location=opt.gpu))
    print('Loaded checkpoint successfully.')
    encoder.eval()
    # print(encoder)
    label_tensor = []
    outputs = []
    with torch.no_grad():
        images = test_dataset[0]
        labels = test_dataset[1]

# 遍历数据集
    sum = 0
    loss_all = 0
    for i in range(len(images)):
        # for image, label in test_loader:
            image = images[i].to(device)
            image = torch.unsqueeze(image, dim=0)
            label = labels[i].to(device)
            output = encoder(image)
            # print(output.shape)
            # print(label.shape)
            loss = criterion(output,label.unsqueeze(0).long()).detach().cpu()
            loss_all += loss
            pred = torch.max(output, dim=1)[1]
            if (torch.eq(pred, label)):
                sum +=1
            # sum_num += torch.eq(pred, real_label.to(device)).sum()
            # if i % 1000 == 0:
            #     print(output)
    #         label_tensor.append(label)
    #         outputs.append(output)
    # outputs = torch.cat(outputs, dim=0)
    # print(outputs.shape)
    # # print(label_tensor.shape)

    # # Calculate test accuracy
    # predicted_labels = np.argmax(outputs.cpu().numpy(), axis=1)
    # test_accuracy = np.mean(predicted_labels == label_tensor) * 100
    test_accuracy = sum/len(images)
    loss_final = loss_all/len(images)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Test Loss: {loss_final:.5f}")
    return test_accuracy


acc = main()
print(acc)




    

