import torch
import torch.nn as nn

# 线性回归模型的类
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):  # 前向传播函数
        return self.linear(x)

model = LinearRegression(input_size=1, output_size=1)

criterion = nn.MSELoss()  # 损失函数评价器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 优化器：随机梯度下降

X = torch.Tensor([[1], [2], [3], [4]])
y = torch.Tensor([[2], [4], [6], [8]])

for epoch in range(1000):
    predictions = model(X)
    loss = criterion(predictions, y)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

X_test = torch.Tensor([[5], [6], [7], [8]])
y_test = torch.Tensor([[10], [12], [14], [16]])


with torch.no_grad():
    predictions = model(X_test)
    loss = criterion(predictions, y_test)
    print(f'Test loss: {loss:.4f}')