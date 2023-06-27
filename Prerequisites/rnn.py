import torch
import torch.nn as nn

rnn = nn.RNN(10, 20, 1, batch_first=True)  # 实例化一个单向单层RNN，带有一个隐含层

input = torch.randn(5, 3, 10)  # 输入序列的形状为(5, 3, 10)，其中5是样本数量，3是batch_size即3个steps时间步，10是每个时间步输入特征维度

h0 = torch.randn(1, 5, 20)

output, hn = rnn(input, h0)  # 调用RNN的forward()函数，得到输出output和最后一个时刻的隐状态hn
print(output)  # 输出output的形状为(5, 3, 20)，其中5是样本数量，3是batch_size即3个steps时间步，20是每个时间步输出特征维度
print(hn)  # 输出hn的最终隐藏层形状为(1, 5, 20)，其中1是隐层层数，5是样本数量，20是每个时间步输出特征维度