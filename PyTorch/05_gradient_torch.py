"""
- Prediction: PyTorch Model
- Gradients Computation: Autograd
- Loss Computation: PyTorch Loss
- Parameter Updates: PyTorch Optimizer
"""

import torch

# Linear Regression f = w * x

# f = 2 * x
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)  # 需要知道这个参数的梯度

# model prediction
def forward(x):
    return w * x

# loss = MSE: mean squared error
def loss(y, y_predicted):
    return ((y_predicted - y) ** 2).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
learing_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward()  # dl/dw

    # update weights
    with torch.no_grad():  # 不需要计算梯度
        w -= learing_rate * w.grad
    
    # zero gradients
    w.grad.zero_()

    if epoch % 10 == 0:  # print every step
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')
