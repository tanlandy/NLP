# chainrule
# dz/dx = dz/dy * dy/dx

"""
1. Forward pass: compute loss
2. Compute local gradients
3. Backward pass: compute dLoss/dweight using the Chain Rule

"""

import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

# forward pass and compute the loss
y_hat = w * x
loss = (y_hat - y) ** 2
print(loss)

# backward pass
loss.backward()  # the whole gradient computation
print(w.grad)  # first gradient 

# update weights
# next forward and backward pass
