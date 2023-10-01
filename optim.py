import torch
import numpy as np
from torch.optim.optimizer import Optimizer, required
from sqrt_mvp import get_inv_corr
import copy

# Base class. Behaves like Adam when used.
# Can be derived to model covariances within a Parameter.
# Override calc_cov and calc_step. 
# Use parameter groups to differentiate between how to


class AdaCovBase(Optimizer):

    def __init__(self, params, lr=0.001, beta_1=0.9,beta_2=0.999,eps=1e-8,num_terms=10,sigma=1.0,delta=1e-4):
        defaults = dict(lr=lr,beta_1=beta_1,beta_2=beta_2,k=0,eps=eps)
        super(AdaCovBase, self).__init__(params, defaults)
        self.beta_2 = beta_2
        self.k=0
        self.num_terms=num_terms
        self.delta=delta
        self.sigma=sigma
        self.reset()
        
    def __setstate__(self, state):
        super(AdaCovBase, self).__setstate__(state)
        
    def update_cov(self,group,param_state, grad):
        param_state["cov"] = group["beta_2"]*param_state["cov"]+ (1.0-group["beta_2"])*grad*grad  
    def calc_step(self,group,param_state,m,cov,diag):
        return m/torch.sqrt(cov+group["eps"]**2)
            
        
    def reset(self):
        self.k = 0
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["mom"] = torch.zeros_like(p.data)
                
                
                if group["cov"]==1:
                    sh = p.data.size()
                    xx=torch.zeros_like(p.data[0,0]).reshape(1,1,sh[2]*sh[3],1)
                    cov = torch.matmul(xx,torch.transpose(xx,-1,-2))
                    param_state["cov"] = cov
                    param_state["diag"] = torch.zeros_like(p.data)
                    assert(sh[2]==sh[3])
                    param_state["Kinvh"] = torch.zeros_like(cov[0,0])
                    get_inv_corr(sh[2],param_state["Kinvh"],sigma=self.sigma,delta=self.delta)
                    param_state["Kinvh"]=param_state["Kinvh"].reshape(1,1,sh[2]*sh[3],sh[2]*sh[3])
                    param_state["Kinvh"]= param_state["Kinvh"]+torch.zeros_like(cov)
                else:
                    param_state["cov"] = torch.zeros_like(p.data)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()
        
        
        for group in self.param_groups:
            beta_1 = group['beta_1']
            beta_2 = group['beta_2']
            lr = group['lr']
            eps = group['eps']
            group["k"]= group["k"] + 1
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                g=p.grad
                param_state = self.state[p]
                
                torch.add(param_state["mom"]*beta_1,g,alpha=1.0-beta_1,out=param_state["mom"]) #First order momen
                self.update_cov(group,param_state,g) #Second order moment
                
                vub   =  param_state["mom"]/(1.0-beta_1**group["k"])
                covub = param_state["cov"]/(1.0-beta_2**group["k"])
                if group["cov"]==1:
                    torch.addcmul(beta_2*param_state["diag"], g,g,value=1.0-beta_2, out=param_state["diag"])
                    diagub = param_state["diag"]/(1.0-beta_2**group["k"])
                else:
                    diagub = None
                
                gg = self.calc_step(group,param_state,vub,covub,diagub)

                p.data = p.data - lr*gg
        
        return loss
   

# Includes Covariances between individual convolutional filters.
# I.e. [n_out,n_in,W,H] becomes n_out x n_in covariance matrices of size [WxH,WxH]   
class AdaConv(AdaCovBase):

    def update_cov(self,group,param_state,grad):
        if group["cov"]==1:
            sh=grad.data.size()
            
        else:
            torch.addcmul(group["beta_2"]*param_state["cov"], grad,grad,value=(1.0-group["beta_2"]), out=param_state["cov"])
        
    def calc_step(self,group,param_state,m,cov,diag):
        if group["cov"]==1:
        
            if "eye" in param_state.keys():
                idm = param_state["eye"]
            else:
                idm = torch.zeros_like(cov[0,0,:,:])
                idm = idm.fill_diagonal_(1.0)
                param_state["eye"]=idm # IS THIS CALL-BY-REFERENCE OK?
        
        
            sh=m.data.size()
 
            
            
            gx = m.reshape(sh[0],sh[1],sh[2]*sh[3],1)
            d = diag.reshape(sh[0],sh[1],sh[2]*sh[3],1)+1e-10#group["eps"]**2

            
            d_inv_sqrt = d**-0.5
            d_qrt = d**0.25
    
            
            gg = torch.matmul(param_state["Kinvh"], gx/d_qrt)/d_qrt
            gg = gg.reshape(sh[0],sh[1],sh[2],sh[3])

            
            return gg
            
        else:
            return m/torch.sqrt(cov+group["eps"]**2)
    

