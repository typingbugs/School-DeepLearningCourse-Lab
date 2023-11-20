import torch
from utils import *


# 手动实现torch.optim.SGD
class My_SGD:
    def __init__(self, params: list[torch.Tensor], lr: float, weight_decay=0.0, momentum=0.0):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.velocities = [torch.zeros_like(param.data) for param in params]

    def step(self):
        with torch.no_grad():
            for index, param in enumerate(self.params):
                if param.grad is not None:
                    if self.weight_decay > 0:
                        if len(param.data.shape) > 1:
                            param.grad.data = (param.grad.data + self.weight_decay * param.data)
                    self.velocities[index] = (self.momentum * self.velocities[index] - self.lr * param.grad)
                    param.data = param.data + self.velocities[index]

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.data = torch.zeros_like(param.grad.data)


if __name__ == "__main__":
    params1 = torch.tensor([[1.0, 2.0]], requires_grad=True)
    params2 = torch.tensor([[1.0, 2.0]], requires_grad=True)

    my_sgd = My_SGD(params=[params1], lr=0.5, momentum=1)
    optim_sgd = torch.optim.SGD(params=[params2], lr=0.5, momentum=1)
    my_sgd.zero_grad()
    optim_sgd.zero_grad()

    loss1 = 2 * params1.sum()
    loss2 = 2 * params2.sum()
    # 偏导为2
    loss1.backward()
    loss2.backward()
    my_sgd.step()
    optim_sgd.step()
    # 结果为：w - lr * grad + momentum * velocity
    # w[0] = 1 - 0.5 * 2 + 1 * 0 = 0
    # w[1] = 2 - 0.5 * 2 + 1 * 0 = 1
    print("My_SGD第1次反向传播结果：\n", params1.data)
    print("torch.optim.SGD第1次反向传播结果：\n", params2.data)

    my_sgd.zero_grad()
    optim_sgd.zero_grad()
    loss1 = -3 * params1.sum()
    loss2 = -3 * params2.sum()
    loss1.backward()
    loss2.backward()
    my_sgd.step()
    optim_sgd.step()
    # 结果为：w - lr * grad + momentum * velocity
    # w[0] = 0 - 0.5 * -3 + 1 * (-0.5 * 2) = 0.5
    # w[1] = 1 - 0.5 * -3 + 1 * (-0.5 * 2) = 1.5
    print("My_SGD第2次反向传播结果：\n", params1.data)
    print("torch.optim.SGD第2次反向传播结果：\n", params2.data)
