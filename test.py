

import torch
import torchvision as tv
from optim import AdaConv
#torch.set_default_dtype(torch.float64)
from model import Net

def run(n_epochs=10,batch_size=128,maxiter=None,fn="out.txt",sdprop=False):
    torch.manual_seed(0)
    
    
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available and being used")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU instead")
        
        
    nc=100
    CTX=torch.device("cuda")
    
    model = Net(3,nc)
    model = model.to(CTX)

    train_data = tv.datasets.CIFAR100("data",train=True,download=True,
            transform = tv.transforms.Compose([tv.transforms.ToTensor()]))

    
    train_data_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=False)
    
    
    
    loss = torch.nn.CrossEntropyLoss()
    
    if sdprop == 2:
        optimizer = AdaConv(model.get_params(),lr=1e-3, beta_1 = 0.9,beta_2=0.999,eps=1e-8)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=0.0)
    k=0
    done=False
    f=open(fn,"w")
    for j in range(0,n_epochs):
        for i,data in enumerate(train_data_loader):
            train_features, train_labels = data
            train_features = train_features.to(CTX)
            train_labels = train_labels.to(CTX)
            optimizer.zero_grad()
            yp = model.forward(train_features)
            err = loss(yp,train_labels)
            err.backward()
            
            ypv,ypid = torch.max(yp,1)
            strr="{0:};{1:};{2:04f};{3:}".format(j,k,err.item(),k*batch_size)
            f.write(strr+"\n")
            #if k %100==0:
            print(strr)
            optimizer.step()
            k+=1
            if  maxiter is not None and k >= maxiter:
                done=True
                break
        if done:
            break
    
    f.close()
    
    #test_data = tv.datasets.CIFAR100("data",train=False,download=True,
    #        transform = tv.transforms.Compose([tv.transforms.ToTensor()]))

    
    test_data_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=False)
    acc,n=0.0,0
    for i,data in enumerate(test_data_loader):
            test_features, test_labels = data
            test_features = test_features.to(CTX)
            test_labels = test_labels.to(CTX)
            yp = model.forward(test_features)
            
            acc += torch.sum(test_labels==torch.argmax(yp,dim=1)).item()
            n+=yp.size()[0]
    print("Test Accuracy: ", acc/n)
if __name__=="__main__":
    import time
    t = time.time()
    run(n_epochs=20, batch_size=128, maxiter=None, fn="covprop_sh.txt", sdprop=2)
    print(time.time()-t)
    
    #t = time.time()
    #run(n_epochs=3, batch_size=128,  maxiter=None, fn="sdprop_sh.txt", sdprop=2)
    #print(time.time()-t)
    
    t = time.time()
    run(n_epochs=20, batch_size=128, maxiter=None, fn="adam_sh.txt", sdprop=0)
    print(time.time()-t)