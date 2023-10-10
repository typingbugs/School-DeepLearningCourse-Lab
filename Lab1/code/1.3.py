import torch

x = torch.tensor(1.0, requires_grad=True)
y_1 = x**2
with torch.no_grad():
    y_2 = x**3

y3 = y_1 + y_2

y3.backward()

print("梯度(dy_3/dx): ", x.grad.item())
