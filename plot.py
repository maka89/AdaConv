import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ma(x,n=10):
    x2 = np.concatenate((x,x[::-1]))
    f = np.zeros_like(x2)
    f[0:n] = np.ones(n)/n
    
   
    m = len(x2)
    X=np.fft.rfft(x2)
    F=np.fft.rfft(f)
    return np.fft.irfft(X*F,n=m)[0:len(x)]

trans = lambda x: ma(x,10)

ipe = 7820/20

fig, (ax1, ax2) = plt.subplots(1, 2)


X = pd.read_csv("covprop_sh.txt",sep=";",skiprows=2).values
ax1.plot(X[:,1]/ipe,trans(X[:,2]))

X = pd.read_csv("adam_sh.txt",sep=";",skiprows=2).values
ax1.plot(X[:,1]/ipe,trans(X[:,2]))
ax1.legend(["AdaConv", "Adam"],fontsize=14)
ax1.set_xlabel("N epochs",fontsize=14)
ax1.set_ylabel("Loss",fontsize=14)
ax1.set_title("3x3 filters",fontsize=14)

X = pd.read_csv("covprop_sh_5x5.txt",sep=";",skiprows=2).values
ax2.plot(X[:,1]/ipe,trans(X[:,2]))

X = pd.read_csv("adam_sh_5x5.txt",sep=";",skiprows=2).values
ax2.plot(X[:,1]/ipe,trans(X[:,2]))
ax2.legend(["AdaConv", "Adam"],fontsize=14)
ax2.set_xlabel("N epochs",fontsize=14)
ax2.set_title("5x5 filters",fontsize=14)
#ax2.set_yticklabels([])
ax2.set_ylim([0,5])
ax1.set_ylim([0,5])



plt.show()