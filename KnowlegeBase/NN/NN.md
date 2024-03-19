# NN

## Activation Function激活函数

[Source](https://www.ai-contentlab.com/2023/03/swishglu-activation-function.html)

激活函数让神经网络实现非线性的功能

### ReLU

### Swish

Swish(x) = x * sigmoid(beta x)

beta is a trainable param

Proposed by Google Researches in 2017

在深度神经网络中，比ReLU更加平滑，优化效果更好，更快收敛

### GLU

Gated Linear Units

GLU(x) = x * sigmoid(Wx + b)

W, b are trainable params

Proposed by Microsoft Researches in 2016

it combines a linear function with a non-linear function, the linear function is gated by a sigmoid activation function.

### SwiGLU

SwiGLU(x) = x \* sigmoid(beta x) + (1 - sigmoid(beta x)) \* (Wx + b)

beta, W, b are trainable params

优点：

1. 更顺滑，更快收敛
2. 非单调性
3. 门机制：减少过拟合，增加泛化能力
