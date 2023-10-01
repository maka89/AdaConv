import torch
import numpy as np
def get_inv_corr(m,out,delta=1e-10,sigma = 1.0):
    x=np.zeros((m,m))
    y=np.zeros((m,m))
    for i in range(0,m):
        for j in range(0,m):
            x[i,j] = j
            y[i,j] = i

    x = x.ravel()
    y = y.ravel()
    K=np.zeros((m*m,m*m))
    for i in range(0,m*m):
        for  j in range(0,m*m):
            dx=(x[i]-x[j])
            dy=(y[i]-y[j])
            K[i,j] = np.exp(-0.5*np.sqrt((dx/sigma)**2+(dy/sigma)**2))
        K[i,i] = K[i,i]+delta
    
    w,q = np.linalg.eigh(K)
    Kinvh = q@np.diag(w**-0.5)@q.T
    for i in range(0,m*m):
        for  j in range(0,m*m):
            out[i,j]=Kinvh[i,j]

            
def pow_series_mv_t(t,C,v):
    n=len(t)
    
    sh=v.size()
    v=v.reshape(sh[0]*sh[1],sh[2],sh[3])
    C=C.reshape(sh[0]*sh[1],sh[2],sh[2])
    f=torch.clone(v)
    for i in range(1,n):
        torch.baddbmm(v,C,v,alpha=-1,out=v)
        torch.add(f,v,alpha=t[i],out=f)
    return f.reshape(sh[0],sh[1],sh[2],sh[3])

def pow_seriest(t,C,eye):
    n=len(t)    
    assert(t[0]==1.0)
    tmp = (eye-C)
    f = eye + t[1]*tmp
    
    for i in range(2,n):
        tmp=tmp-tmp@C
        f+= t[i]*tmp
        
    return f
    
def ps_coefs(n=10,inv=False):
    
    if inv:
        a=-0.5
    else:
        a=0.5
    
    binom = 1.0
    t=[binom]
    for i in range(1,n):
        binom*= (a+1-i)/i
        t.append(((-1)**i)*binom)
    return t
    

def pade_series(n=5,inverse=False):
    b=np.array(ps_coefs(n*2-1,inv=inverse))
    A=np.zeros((2*n-1,2*n-1))
    A[0:n,0:n] = np.eye(n)
    
    for i in range(0,n-1):
        A[(i+1)::,n+i] = -b[0:-(i+1)]

    x=np.linalg.solve(A,b)
    a = x[0:n]
    b = x[n::]
    b = np.concatenate([[1.0],b])
    
    return a,b
    
class Pade:
    def __init__(self,n_terms,inverse):
        self.a,self.b = pade_series(n_terms,inverse)
        assert(self.a[0]==1.0)
        assert(self.b[0]==1.0)
        self.inverse = inverse
        self.eye=None
        
        
    def set_mat(self,C):
        if self.eye is None or self.eye.size()!=C.size():
            idm = torch.zeros_like((C.reshape(-1,C.size()[-2],C.size()[-1]))[0])
            idm = idm.fill_diagonal_(1.0)
            self.eye=torch.zeros_like(C)+idm   

        self.scale = torch.sqrt(torch.sum(C**2,dim=(-2,-1),keepdims=True))
        self.C = C/self.scale
        self.Q=pow_seriest(self.b,self.C,self.eye)
        self.sqrt_scale = torch.sqrt(self.scale)
    def matvec(self,x):
        u = pow_series_mv_t(self.a,self.C,x)
        u = torch.linalg.solve(self.Q,u)
        
        if self.inverse:
            return u/self.sqrt_scale
        else:
            return u*self.sqrt_scale
            
class Taylor:
    def __init__(self,n_terms,inverse):
        self.t = ps_coefs(n_terms,inverse)
        self.inverse = inverse
        self.eye=None
        
    def set_mat(self,C):
        if self.eye is None or self.eye.size()!=C.size():
            idm = torch.zeros_like((C.reshape(-1,C.size()[-2],C.size()[-1]))[0])
            idm = idm.fill_diagonal_(1.0)
            self.eye=torch.zeros_like(C)+idm    
        
        self.scale = torch.sqrt(torch.sum(C**2,dim=(-2,-1),keepdims=True))
        self.C = C/self.scale
        self.sqrt_scale = torch.sqrt(self.scale)
    def matvec(self,x):
        u = pow_series_mv_t(self.t,self.C,x)
        if self.inverse:
            return u/self.sqrt_scale
        else:
            return u*self.sqrt_scale
        

if __name__=="__main__":
    import time
    def sqrt(C,inv=False):
        w,q = torch.linalg.eigh(C)
        w = w[...,None]
        if inv:
            a=-0.5
        else:
            a=0.5
        return  torch.matmul(q,w**a*torch.transpose(q,-1,-2))
    
    #torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    N=50
    inv = False
    X=torch.randn(100,N)
    v=torch.randn(N,1)
    C=torch.matmul(torch.transpose(X,-1,-2),X)
    
    p = Pade(6,inv)
    t = Taylor(20,inv)
    t.set_mat(C)
    sq1 = sqrt(C,inv)
    
    p.set_mat(C)

    pv = p.matvec(v)

    print("Pade: ",torch.mean(torch.linalg.norm(sq1@v-pv,dim=(-2,-1)) ))
    print("Taylor: ", torch.mean(torch.linalg.norm(sq1@v-t.matvec(v),dim=(-2,-1)) ))
    
    from torch_utils import *
    FastMatSqrt = MPA_Lya.apply
    FastInvSqrt = MPA_Lya_Inv.apply
    if inv:
        sq2 = FastInvSqrt(C)
    else:
        sq2 = FastMatSqrt(C)
    print("Pade FastDiffSqrt: ",torch.mean(torch.linalg.norm(sq2@v-sq1@v,dim=(-2,-1)) ))
    print("Pade FastDiffSqrt - Pade : ",torch.mean(torch.linalg.norm(sq2@v-p.matvec(v),dim=(-2,-1)) ))
    