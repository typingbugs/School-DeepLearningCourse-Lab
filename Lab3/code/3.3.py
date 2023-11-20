import torch
from utils import *


class My_Adam:
    def __init__(self, params: list[torch.Tensor], lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.t = 0
        self.momentums = [torch.zeros_like(param.data) for param in params]
        self.velocities = [torch.zeros_like(param.data) for param in params]

    def step(self):
        self.t += 1
        with torch.no_grad():
            for index, param in enumerate(self.params):
                if param.grad is not None:
                    self.momentums[index] = (self.beta1 * self.momentums[index] + (1 - self.beta1) * param.grad)
                    self.velocities[index] = (self.beta2 * self.velocities[index] + (1 - self.beta2) * param.grad ** 2)

                    momentums_hat = self.momentums[index] / (1 - self.beta1 ** self.t)
                    velocities_hat = self.velocities[index] / (1 - self.beta2 ** self.t)

                    param.data = param.data - self.lr * momentums_hat / (torch.sqrt(velocities_hat) + self.eps)

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.data = torch.zeros_like(param.grad.data)


if __name__ == "__main__":
    params1 = torch.tensor([[1.0, 2.0]], requires_grad=True)
    params2 = torch.tensor([[1.0, 2.0]], requires_grad=True)

    my_sgd = My_Adam(params=[params1], lr=1, betas=(0.5, 0.5), eps=1e-8)
    optim_sgd = torch.optim.Adam(params=[params2], lr=1, betas=(0.5, 0.5), eps=1e-8)
    my_sgd.zero_grad()
    optim_sgd.zero_grad()

    loss1 = 2 * params1.sum()
    loss2 = 2 * params2.sum()
    # 偏导为2
    loss1.backward()
    loss2.backward()
    my_sgd.step()
    optim_sgd.step()
    print("My_Adam第1次反向传播结果：\n", params1.data)
    print("torch.optim.Adam第1次反向传播结果：\n", params2.data)

    my_sgd.zero_grad()
    optim_sgd.zero_grad()
    loss1 = -3 * params1.sum()
    loss2 = -3 * params2.sum()
    # 偏导为-3
    loss1.backward()
    loss2.backward()
    my_sgd.step()
    optim_sgd.step()
    print("My_Adam第2次反向传播结果：\n", params1.data)
    print("torch.optim.Adam第2次反向传播结果：\n", params2.data)
