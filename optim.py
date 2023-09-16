import torch
import numpy as np
from torch.optim.optimizer import Optimizer, required
from utils import torch_refSyEv
import copy

# Base class. Behaves like Adam when used.
# Can be derived to model covariances within a Parameter.
# Override calc_cov and calc_step. 
# Use parameter groups to differentiate between how to


class AdaCov(Optimizer):

    def __init__(self, params, lr=0.001, beta_1=0.9,beta_2=0.999,eps=1e-8):
        defaults = dict(lr=lr,beta_1=beta_1,beta_2=beta_2,k=0,eps=eps)
        super(AdaCov, self).__init__(params, defaults)
        self.beta_2 = beta_2
        self.k=0
        self.reset()
        
    def __setstate__(self, state):
        super(AdaCov, self).__setstate__(state)
        
    def calc_cov(self,group, grad):
        return grad*grad
            
    def calc_step(self,group,param_state,m,cov):
        return m/torch.sqrt(cov+group["eps"]**2)
            
        
    def reset(self):
        self.k = 0
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["mom"] = torch.zeros_like(p.data)
                param_state["cov"] = self.calc_cov(group,torch.zeros_like(p.data))
                

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
                
                
                v = param_state['mom']
                cov = param_state["cov"]
                
                v = beta_1*v + (1.0-beta_1)*g
                vub   = v/(1.0-beta_1**group["k"])
                
                cov = beta_2*cov + (1.0-beta_2)*self.calc_cov(group,g)
                covub = cov/(1.0-beta_2**group["k"])
                
                param_state["mom"] = v
                param_state["cov"] = cov
                
                gg = self.calc_step(group,param_state,vub,covub)

                p.data = p.data - lr*gg
        
        return loss
   

# Includes Covariances between individual convolutional filters.
# I.e. [n_out,n_in,W,H] becomes n_out x n_in covariance matrices of size [WxH,WxH]   
class AdaConv(AdaCov):

    def calc_cov(self,group, grad):
        if group["cov"]==1:
            sh=grad.data.size()
            xx=grad.reshape(sh[0],sh[1],sh[2]*sh[3],1)
            return torch.matmul(xx,torch.transpose(xx,3,2))
        else:
            return grad*grad
        
    def calc_step(self,group,param_state,m,cov):
        if group["cov"]==1:
        
            if "eye" in param_state.keys():
                idm = param_state["eye"]
            else:
                idm = torch.zeros_like(cov[0,0,:,:])
                idm = idm.fill_diagonal_(1.0)
                param_state["eye"]=idm # IS THIS CALL-BY-REFERENCE OK?
        
        
            sh=m.data.size()
 
            covd = cov*idm
            cov1 = cov
            
            
            a = 10.0**(-(group["k"]/300))
            cov2 = covd*a+(1.0-a)*cov1
            
            init_diag = np.log10(group["eps"]**2)
            final_diag = -10
            add_diag = init_diag + (final_diag-init_diag)*(1.0-np.exp(-0.5*(group["k"]/300)**2))
            add_diag = 10.0**add_diag
            add_diag = max(add_diag,final_diag)
            
            cov2 = cov2 + idm*add_diag
            
            w,q = torch.linalg.eigh(cov2)

            w = w.reshape(sh[0],sh[1],sh[2]*sh[3],1)
            
            gx = m.reshape(sh[0],sh[1],sh[2]*sh[3],1)
            gg = w**-0.5*torch.matmul(torch.transpose(q,3,2),gx)
            gg = torch.matmul(q,gg)
            gg = gg.reshape(sh[0],sh[1],sh[2],sh[3])

            
            return gg
            
        else:
            return m/torch.sqrt(cov+group["eps"]**2)
    

