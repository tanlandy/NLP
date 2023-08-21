"""
Typical training pipeline in PyTorch:

1 ) Design model (input, output size, forward pass)
2 ) Construct loss and optimizer
3 ) Training loop
    - forward pass: compute prediction
    - backward pass: gradients
    - update weights
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler  # scale features
from sklearn.model_selection import train_test_split  # split dataset

# 0) prepare data
bc = datasets.load_breast_cancer()  # binary classification dataset
X, y = bc.data, bc.target

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)  # split dataset

# scale features
sc = StandardScaler()  # 0 mean and unit variance
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# convert to tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)  # reshape tensor to have only one col
y_test = y_test.view(y_test.shape[0], 1)  # reshape tensor to have only one col


# 1) model
# f = wx + b, sigmoid at the end
class LogisticRegression(nn.Module):  # inherit from nn.Module
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)  # output is 1

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted


model = LogisticRegression(n_features)  # n_features = 30


# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss()  # binary cross entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    # backward pass
    loss.backward()

    # updates
    optimizer.step()

    optimizer.zero_grad()  # zero gradients

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()  # round to 0 or 1
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')

