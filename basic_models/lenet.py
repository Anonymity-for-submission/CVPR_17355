import torch
from torch import  nn
import torch.nn.functional as F
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms

class Lenet5(nn.Module):

    def __init__(self,input_channels):
        super().__init__()
        #第1个卷积层
        self.conv1 = nn.Conv2d(input_channels , 6 , kernel_size = 5 , padding = 2)
        #第1个池化层
        self.pooling1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        #第2个卷积层
        self.conv2= nn.Conv2d(6 , 16 , kernel_size=5)
        #第2个池化层
        self.pooling2 = nn.MaxPool2d(kernel_size = 2, stride=2)
        ##最后的三个FC
        self.Flatten = nn.Flatten()
        # 计算得出的当前的前面处理过后的shape，当然也可print出来以后再确定
        self.Linear1 = nn.Linear(16*6*6,120)
        self.Linear2 = nn.Linear(120,84)
        self.Linear3 = nn.Linear(84,10)

    def forward(self,X,if_feature=False):
    
        X = self.pooling1(F.relu(self.conv1(X)))
        X = self.pooling2(F.relu(self.conv2(X)))
        X = X.view(X.size()[0],-1)
        X = F.relu(self.Linear1(X))
        feature = F.relu(self.Linear2(X))
        X = F.relu(self.Linear3(feature))
        if if_feature:
            return X,feature
        return X
