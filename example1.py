import torch
import torch

#x = torch.tensor(3.)
#w= torch.tensor(4., requires_grad=True)
#b = torch.tensor(5., requires_grad=True)

#y = w * x  + b

#print(y)
#print(y.backward())
#print(x.grad) #dy/dx
#print(w.grad) #dy/dw
#print(b.grad) #dy/db


import numpy as np

#temp, rain, humidty
inputs = np.array([[73, 67, 43],[91,88,64],[87,134, 58],[102,43,37],[69,96, 70]],dtype='float32')

#apple, oranges
targets = np.array([[56, 70],[81, 101],[119,133],[22,37],[103,119]],dtype='float32')

inputs_tensor=torch.from_numpy(inputs)
targets_tensor=torch.from_numpy(targets)

w = torch.randn(2,3 ,requires_grad=True)
b = torch.randn(2, requires_grad=True)

print(w,b)

def model(x):
    return x @ w.t() + b

#preds=model(inputs_tensor)
#print(preds)
#print(targets_tensor)

##compute loss
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff*diff)/diff.numel()

#loss = mse(preds,targets_tensor)
#print(loss.backward())

#print(w)
#print(w.grad)

#print(inputs_tensor)
#print(inputs_tensor.grad)

#print(b)
#print(b.grad)

#w.grad.zero_()
#b.grad.zero_()

#print(w.grad)
#print(b.grad)


#training

for i in range(1000):
    preds=model(inputs_tensor)
    loss=mse(preds,targets_tensor)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()

print(loss)

print(preds)
print(targets_tensor)