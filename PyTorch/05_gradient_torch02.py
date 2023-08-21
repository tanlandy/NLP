"""
- Prediction: Manually
- Gradients Computation: Autograd
- Loss Computation: PyTorch Loss
- Parameter Updates: PyTorch Optimizer
"""

"""
Typicall training pipeline in PyTorch: 

1 ) Design model (input, output size, forward pass)
2 ) Construct loss and optimizer
3 ) Training loop
    - forward pass: compute prediction
    - backward pass: gradients
    - update weights
"""

import torch
import torch.nn as nn  # neural network

# Linear Regression f = w * x

# f = 2 * x
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)  # 需要知道这个参数的梯度

# model prediction
def forward(x):
    return w * x

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
learing_rate = 0.01
n_iters = 100

loss = nn.MSELoss()  # mean squared error
optimizer = torch.optim.SGD([w], lr=learing_rate)  # SGD: stochastic gradient descent

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward()  # dl/dw

    # update weights
    optimizer.step()
    
    # zero gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:  # print every step
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')

