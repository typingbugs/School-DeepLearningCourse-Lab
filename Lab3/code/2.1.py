import torch
from utils import *


class My_SGD:
    def __init__(self, params: list[torch.Tensor], lr: float, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        with torch.no_grad():
            for param in self.params:
                if param.grad is not None:
                    if len(param.data.shape) > 1:
                        param.data = param.data - self.lr * (param.grad + self.weight_decay * param.data)
                    else:
                        param.data = param.data - self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.data = torch.zeros_like(param.grad.data)


if __name__ == "__main__":
    params1 = torch.tensor([[1.0, 2.0]], requires_grad=True)
    params2 = torch.tensor([[1.0, 2.0]], requires_grad=True)

    my_sgd = My_SGD(params=[params1], lr=0.5, weight_decay=0.1)
    optim_sgd = torch.optim.SGD(params=[params2], lr=0.5, weight_decay=0.1)
    my_sgd.zero_grad()
    optim_sgd.zero_grad()

    loss1 = 2 * params1.sum()
    loss2 = 2 * params2.sum()
    # 偏导为2
    loss1.backward()
    loss2.backward()
    print("params1的梯度为：\n", params1.grad.data)
    print("params2的梯度为：\n", params2.grad.data)

    my_sgd.step()
    optim_sgd.step()
    # 结果为：w - lr * grad - lr * weight_decay_rate * w
    # w[0] = 1 - 0.5 * 2 - 0.5 * 0.1 * 1 = -0.0500
    # w[1] = 2 - 0.5 * 2 - 0.5 * 0.1 * 2 = 0.9000
    print("经过L_2正则化后的My_SGD反向传播结果：\n", params1.data)
    print("经过L_2正则化后的torch.optim.SGD反向传播结果：\n", params2.data)
