import torch

x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2  # this will create a computation graph
print(y)  # tensor([1.3447, 1.5184, 2.6053], grad_fn=<AddBackward0>)
z = y * y * 2
print(z)  # tensor([ 3.6173,  4.6083, 13.5792], grad_fn=<MulBackward0>)
z = z.mean()
print(z)  # tensor(7.9349, grad_fn=<MeanBackward0>)
z.backward()  # calculate gradient: dz/dx
print(x.grad)

z2 = x * x * 2
v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z2.backward(v)  # calculate gradient: dz/dx, z is not a scalar, so we need to pass a vector to backward()

# stop autograd from tracking history on Tensors
x.requires_grad_(False)  # option 1
print(x)
y = x.detach()  # option 2
print(y)
with torch.no_grad():  # option 3
    y = x + 2
    print(y)

weights = torch.ones(4, requires_grad=True)

for epoch in range(3):  # training loop
    model_output = (weights * 3).sum()

    model_output.backward()

    print(weights.grad)  # print gradient

    weights.grad.zero_()  # zero out the gradient in each epoch

# use a pytorch built-in optimizer, without the need for the training loop
optimizer = torch.optim.SGD(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad()