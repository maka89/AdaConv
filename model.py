import torch
import torch.nn as nn
class Net(torch.nn.Module):
    def __init__(self,img_channels=3,num_classes:int = 100):
        super(Net,self).__init__()
        
        ks = [7,7]
        self.conv1 = torch.nn.Conv2d(img_channels,64,kernel_size=ks,bias=True,padding="same")
        self.conv2 = torch.nn.Conv2d(64,64,kernel_size=ks,bias=True,padding="same")
        
        self.conv3 = torch.nn.Conv2d(64,64,kernel_size=ks,bias=True,padding="same")
        self.conv4 = torch.nn.Conv2d(64,64,kernel_size=ks,bias=True,padding="same")
        
        self.conv5 = torch.nn.Conv2d(64,128,kernel_size=ks,bias=True,padding="same")
        self.conv6 = torch.nn.Conv2d(128,128,kernel_size=ks,bias=True,padding="same")
        
        
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d((2,2))
        
       
        
        
        self.avgpool = torch.nn.AdaptiveMaxPool2d((1,1))
        self.fc = torch.nn.Linear(128,256)
        self.fc2 = torch.nn.Linear(256,num_classes)

        # Mark these parameters for AdaConv optimizer.
        self.cov_nm = ["conv1.weight","conv2.weight","conv3.weight","conv4.weight","conv5.weight","conv6.weight"]
        
    # Split into two parameter groups.
    # Conv. weights with group["cov"]==1 will get their covariances estimated by AdaConv.
    def get_params(self):
        l_vanilla = []
        l_cov = []
        l_cov2 = []
        
        
        for name, param in self.named_parameters():
            
            if name in self.cov_nm:
                l_cov.append(param)
            else:
                l_vanilla.append(param)
        d0={"params":l_vanilla,"cov":0}
        d1={"params":l_cov,"cov":1}
        
        d=[d0,d1]
        print("# Regular parameters: ", len(d0["params"]))
        print("# AdaConv parameters: ", len(d1["params"]))
        return d
    def forward(self,x):
        h = self.conv1(x)
        h = self.relu(h)
        
        h = self.conv2(h)
        h = self.relu(h)
        
        h = self.maxpool(h)
        
        h = self.conv3(h)
        h = self.relu(h)
        
        h = self.conv4(h)
        h = self.relu(h)
        
        h = self.maxpool(h)
        
        h = self.conv5(h)
        h = self.relu(h)
        
        h = self.conv6(h)
        
        h = self.avgpool(h)
        h = torch.flatten(h,1)
        h = self.fc(h)
        h = self.relu(h)
        h = self.fc2(h)
        return h
        
        
        
        
        