"""
- Prediction: PyTorch Model
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


X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

n_samples, n_features = X.shape
print(f'#samples: {n_samples}, #features: {n_features}')

X_test = torch.tensor([5], dtype=torch.float32)
input_size = n_features
output_size = n_features
model = nn.Linear(input_size, output_size)  # 等价于 w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# 接下来几行，等效于上面的 model = nn.Linear(input_size, output_size)
"""
class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size) 
"""

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# Training
learing_rate = 0.01
n_iters = 200

loss = nn.MSELoss()  # mean squared error
optimizer = torch.optim.SGD(model.parameters(), lr=learing_rate)  # SGD: stochastic gradient descent

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward()  # dl/dw

    # update weights
    optimizer.step()
    
    # zero gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:  # print every step
        [w, b] = model.parameters()  # unpack parameters
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')

