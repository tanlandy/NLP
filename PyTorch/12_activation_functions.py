"""
activation functions apply a non-linear transformations and decide whether a neuron should be activated or not.
Without activation functions, the neural network is a stacked linear regression model, would be only able to learn linear relationships.
after each layer, we apply an activation function to the layer’s output to introduce non-linearity into the network.
Popular activation functions are
1. step function (return 1 if x is greater than a threshold not used in practice)
2. sigmoid (used in the output layer for binary classification)
3. tanh (a shifted scaled sigmoid function, good choice for hidden layers)
4. ReLU (most popular, used in hidden layers)
5. Leaky ReLU (improved version of ReLU, tries to solve the vanishing gradient problem)
6. Softmax (used in the output layer for multi-class classification)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# option 1 (create nn modules)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()  # nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU(), nn.Softmax()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out


# option 2 (use activation functions directly in forward pass)
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = torch.relu(self.linear1(x))  # torch.sigmoid(), torch.tanh(), F.leaky_relu()
        out = torch.sigmoid(self.linear2(out))
        return out
