# AdaConv
Adam, but includes covariances in the CNN filters in the covariance matrix.

## Usage
Use as a regular torch optimizer.

```python
optimizer = AdaConv(model.get_params(),lr=1e-3, beta_1 = 0.9,beta_2=0.999,eps=1e-8)
```

Parameters need to be split in two groups, one for the conv. weights you want to model covariances for, and one for the other parameters, whcih will be treated like standard Adam.
Conv weights should be marked by being in a group where group["cov"] = 1.

You can use something like this, instead of model.parameters() to pass your parameters to the optimizer:
```python
def get_params(self):
        l_vanilla = []
        l_cov = []
        l_cov2 = []

        # List of conv. weights. Include covariances within the filters of these weights.
        cov_nm=["conv1.weight","conv2.weight","conv3.weight","conv4.weight","conv5.weight","conv6.weight"]

        for name, param in self.named_parameters():
            
            if name in cov_nm:
                l_cov.append(param)
            else:
                l_vanilla.append(param)
        d0={"params":l_vanilla,"cov":0}
        d1={"params":l_cov,"cov":1}
        
        d=[d0,d1]
        print("# Regular parameters: ", len(d0["params"]))
        print("# AdaConv parameters: ", len(d1["params"]))
        return d
```

