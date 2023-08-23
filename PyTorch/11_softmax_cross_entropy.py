"""
softmax: 将向量变成概率分布
"""

import torch
import torch.nn as nn
import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print('softmax numpy:', outputs)

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)  # dim=0, 按列计算
print('softmax torch:', outputs)

# Cross entropy loss 经常和 softmax 一起使用


def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss  # / float(predicted.shape[0])


# y must be one-hot encoded
# if class 0: [1, 0, 0]
# if class 1: [0, 1, 0]
# if class 2: [0, 0, 1]
y = np.array([1, 0, 0])

# y_pred has probabilities
y_pred_good = np.array([0.7, 0.2, 0.1])
y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(y, y_pred_good)
l2 = cross_entropy(y, y_pred_bad)
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')

# in PyTorch
loss = nn.CrossEntropyLoss()
# 1. No Softmax in last layer, as nn.CrossEntropyLoss applies nn.LogSoftmax + nn.NLLLoss
# 2. Y has class labels, not one-hot
# 3. Y_pred has raw scores (logits), no softmax

Y = torch.tensor([0])  # correct class's label is 0
# nsamples x nclasses = 1 x 3
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print(f'Loss1 PyTorch: {l1.item():.4f}')
print(f'Loss2 PyTorch: {l2.item():.4f}')

# get the predictions
_, predictions1 = torch.max(Y_pred_good, 1)  # align the first dimension
_, predictions2 = torch.max(Y_pred_bad, 1)
print(predictions1)
print(predictions2)

# 3 samples
Y = torch.tensor([2, 0, 1])
# nsamples x nclasses = 3 x 3
Y_pred_good = torch.tensor([[0.1, 1.0, 2.1],
                            [2.0, 1.0, 0.1],
                            [0.1, 3.0, 0.1]])
Y_pred_bad = torch.tensor([[2.1, 1.0, 0.1],
                            [0.1, 1.0, 2.1],
                            [0.1, 3.0, 0.1]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print(f'Loss1 PyTorch: {l1.item():.4f}')
print(f'Loss2 PyTorch: {l2.item():.4f}')

# get the predictions
_, predictions1 = torch.max(Y_pred_good, 1)  # align the first dimension
_, predictions2 = torch.max(Y_pred_bad, 1)
print(predictions1)
print(predictions2)