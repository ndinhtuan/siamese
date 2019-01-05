import torch.nn as nn 
import torch.functional as F 
import torch

class Siamese(nn.Module):

    def __init__(self):
        
        super(Siamese, self).__init__()
        #input : 56x46
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=15, kernel_size=7, stride=1, padding=0)#15, 7x7, 1 -> 50x40x15
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)#2x2 -> 25x20
        self.conv2 = nn.Conv2d(15, 45, 6, 1)#45, 6x6, 1 -> 20x15x45
        self.pool2 = nn.MaxPool2d((4, 3), (4, 3))#4x3 -> 5x5
        self.conv3 = nn.Conv2d(45, 250, 5, 1)#250, 5x5 -> 1x1x250

        self.fully = nn.Linear(250, 100)#num_unit=50
        self.linear1 = nn.Linear(100, 50)
        self.linear2 = nn.Linear(50, 2)
        
        self.feature_map = nn.Sequential(
                self.conv1,
                nn.ReLU(),
                self.pool1,
                self.conv2,
                nn.ReLU(),
                self.pool2,
                self.conv3,
                nn.ReLU()
                )

    def forward1(self, x):
        x = torch.unsqueeze(x,1)
        #print(x.size(), x.dtype)
        #x = x.type(torch.FloatTensor)
        x = self.conv1(x)
        x = nn.ReLU()(x) 
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = x.view(x.size(0), 250)
        #x = nn.Dropout()(x)
        x = self.fully(x)
        x = self.linear1(x)
        return x

    def forward(self, x1, x2):
        f1 = self.forward1(x1) 
        f2 = self.forward1(x2) 
        dis = torch.abs(f1-f2)
        res = self.linear2(dis)
        return nn.Softmax(dim=1)(res)


import numpy as np 

class SiameseLoss(nn.Module):

    def __init__(self):
        super(SiameseLoss, self).__init__() 
        self.Q = 3.0
    
    #def forward(self, f1, f2, y):

    def forward(self, f1, f2, y):
        
        ret = torch.abs(f1-f2)
        ret = torch.mul(ret, ret)
        #print(ret)
        ret = torch.sum(ret, 1).type(torch.FloatTensor)**(1.0/2)
        print(ret, y)
        #print(ret.size(), ret.dtype)
        #print(torch.mul((1-y),(2.0/self.Q)).dtype)
        #print(torch.mul(ret,ret).dtype)
        a = torch.mul(torch.mul((1-y.type(torch.FloatTensor)),(2.0/self.Q)).type(torch.FloatTensor), (torch.mul(ret,ret)))
        #print torch.mul(ret,ret), torch.mul((1-y),(2.0/self.Q)), 1-y
        b = y.type(torch.FloatTensor)*(2*self.Q*torch.exp(-2.77*ret/self.Q))
        ret = torch.sum(a + b)
        return ret 

