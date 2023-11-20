import torch
from utils import *


class My_RMSprop:
    def __init__(self, params: list[torch.Tensor], lr=1e-2, alpha=0.99, eps=1e-8):
        self.params = params
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.square_avg = [torch.zeros_like(param.data) for param in params]
        

    def step(self):
        with torch.no_grad():
            for index, param in enumerate(self.params):
                if param.grad is not None:
                    self.square_avg[index] = self.alpha * self.square_avg[index] + (1 - self.alpha) * param.grad ** 2
                    param.data = param.data - self.lr * param.grad / torch.sqrt(self.square_avg[index] + self.eps)

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.data = torch.zeros_like(param.grad.data)


if __name__ == "__main__":
    params1 = torch.tensor([[1.0, 2.0]], requires_grad=True)
    params2 = torch.tensor([[1.0, 2.0]], requires_grad=True)

    my_sgd = My_RMSprop(params=[params1], lr=1, alpha=0.5, eps=1e-8)
    optim_sgd = torch.optim.RMSprop(params=[params2], lr=1, alpha=0.5, eps=1e-8)
    my_sgd.zero_grad()
    optim_sgd.zero_grad()

    loss1 = 2 * params1.sum()
    loss2 = 2 * params2.sum()
    # 偏导为2
    loss1.backward()
    loss2.backward()
    my_sgd.step()
    optim_sgd.step()
    # s = alpha * s + (1-alpha) * grad^2 = 0.5 * 0 + (1-0.5) * 2^2 = 2
    # w = w - lr * grad * (s + eps)^0.5
    # w[0] = 1 - 1 * 2 / (2 + 1e-8)^0.5 ~= -0.41
    # w[1] = 2 - 1 * 2 / (2 + 1e-8)^0.5 ~= -0.59
    print("My_RMSprop第1次反向传播结果：\n", params1.data)
    print("torch.optim.RMSprop第1次反向传播结果：\n", params2.data)

    my_sgd.zero_grad()
    optim_sgd.zero_grad()
    loss1 = -3 * params1.sum()
    loss2 = -3 * params2.sum()
    loss1.backward()
    loss2.backward()
    my_sgd.step()
    optim_sgd.step()
    # s = alpha * s + (1-alpha) * grad^2 = 0.5 * 2 + (1-0.5) * (-3)^2 = 5.5
    # w - lr * grad * (s + eps)^0.5
    # w[0] = -0.41 - 1 * -3 / (5.5 + 1e-8)^0.5 ~= 0.87
    # w[1] = 0.59 - 1 * -3 / (5.5 + 1e-8)^0.5 ~= 1.86
    print("My_RMSprop第2次反向传播结果：\n", params1.data)
    print("torch.optim.RMSprop第2次反向传播结果：\n", params2.data)