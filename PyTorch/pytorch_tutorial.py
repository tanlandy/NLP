import torch
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")  # a CUDA device object
    x = torch.ones(5, device=device)  # directly create a tensor on GPU
    y = torch.ones(5)  # create a tensor on CPU
    y = y.to(device)  # move y to GPU
    z = x + y  # add two tensors on GPU
    z = z.to("cpu")  # move z to CPU

x = torch.empty(2, 3)

print(x)

y = torch.rand(2, 2, dtype=torch.double)
print(y.dtype)

y.add_(x)  # in place add
z = x - y
z = torch.sub(x, y)  # same as the line above

z = x * y
z = torch.mul(x, y)  # same as the line above

z = x / y
z = torch.div(x, y)  # same as the line above

x = torch.rand(5, 3)
print(x[:, 0])  # all rows, cols[0]
print(x[1, 1])  # element at [1][1]
print(x[1, 1].item())  # get the actural value of the element

# reshape a tensor
x = torch.rand(4, 4)
y = x.view(16)  # one dimention now, 16 = 4 * 4
y = x.view(-1, 8)  # two by eight tensor, the size will be [2, 8]

# numpy and tensor
# numpy can only handle CPU tensor
a = torch.ones(5)
b = a.numpy()  # tensor to numpy
# if a and b are stored in CPU, then if we change b, a will be changed, as they share the same memory location
c = np.ones(5)
d = torch.from_numpy(c)  # numpy to tensor

# requires_grad
x = torch.ones(5, requires_grad=True)  # this tensor will be used to calculate gradient later

