import time
import numpy as np
import torch
from torch.nn.functional import *
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import datasets, transforms
from tqdm import tqdm

import ipdb


# 手动实现torch.nn.functional.one_hot
def my_one_hot(indices: torch.Tensor, num_classes: int):
    one_hot_tensor = torch.zeros(len(indices), num_classes, dtype=torch.long).to(indices.device)
    one_hot_tensor.scatter_(1, indices.view(-1, 1), 1)
    return one_hot_tensor


# 手动实现torch.nn.functional.softmax
def my_softmax(predictions: torch.Tensor, dim: int):
    max_values = torch.max(predictions, dim=dim, keepdim=True).values
    exp_values = torch.exp(predictions - max_values)
    softmax_output = exp_values / torch.sum(exp_values, dim=dim, keepdim=True)
    return softmax_output

# 手动实现torch.nn.Linear
class My_Linear:
    def __init__(self, in_features: int, out_features: int):
        self.weight = torch.normal(mean=0.001, std=0.5, size=(out_features, in_features), requires_grad=True, dtype=torch.float32)
        self.bias = torch.normal(mean=0.001, std=0.5, size=(1,), requires_grad=True, dtype=torch.float32)
        self.params = [self.weight, self.bias]

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = torch.matmul(x, self.weight.T) + self.bias
        return x

    def to(self, device: str):
        for param in self.params:
            param.data = param.data.to(device=device)
        return self

    def parameters(self):
        return self.params

        
# 手动实现torch.nn.Flatten
class My_Flatten:
    def __call__(self, x: torch.Tensor):
        x = x.view(x.shape[0], -1)
        return x

        
# 手动实现torch.nn.ReLU
class My_ReLU():
    def __call__(self, x: torch.Tensor):
        x = torch.max(x, torch.tensor(0.0, device=x.device))
        return x


# 手动实现torch.nn.Sigmoid
class My_Sigmoid():
    def __call__(self, x: torch.Tensor):
        x = 1. / (1. + torch.exp(-x))
        return x


# 手动实现torch.nn.BCELoss
class My_BCELoss:
    def __call__(self, prediction: torch.Tensor, target: torch.Tensor):
        loss = -torch.mean(target * torch.log(prediction) + (1 - target) * torch.log(1 - prediction))
        return loss


# 手动实现torch.nn.CrossEntropyLoss
class My_CrossEntropyLoss:
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor):
        max_values = torch.max(predictions, dim=1, keepdim=True).values
        exp_values = torch.exp(predictions - max_values)
        softmax_output = exp_values / torch.sum(exp_values, dim=1, keepdim=True)

        log_probs = torch.log(softmax_output)
        nll_loss = -torch.sum(targets * log_probs, dim=1)
        average_loss = torch.mean(nll_loss)
        return average_loss


# 手动实现torch.optim.SGD
class My_optimizer:
    def __init__(self, params: list[torch.Tensor], lr: float):
        self.params = params
        self.lr = lr

    def step(self):
        with torch.no_grad():
            for param in self.params:
                param.data = param.data - self.lr * param.grad.data

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.data = torch.zeros_like(param.grad.data)


def train_MNIST_CLS(Model:nn.Module):
    learning_rate = 8e-2
    num_epochs = 10
    batch_size = 512
    num_classes = 10
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_mnist_dataset = datasets.MNIST(root="../dataset", train=True, transform=transform, download=True)
    test_mnist_dataset = datasets.MNIST(root="../dataset", train=False, transform=transform, download=True)
    train_loader = DataLoader(
        dataset=train_mnist_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=14,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset=test_mnist_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=14,
        pin_memory=True,
    )

    model = Model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    train_loss = list()
    test_acc = list()
    for epoch in range(num_epochs):
        model.train()
        total_epoch_loss = 0
        start_time = time.time()
        for index, (images, targets) in tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):
            optimizer.zero_grad()

            images = images.to(device)
            targets = targets.to(device)
            one_hot_targets = one_hot(targets, num_classes=num_classes).to(dtype=torch.float)

            outputs = model(images)
            loss = criterion(outputs, one_hot_targets)
            total_epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        end_time = time.time()
        train_time = end_time - start_time

        model.eval()
        with torch.no_grad():
            total_epoch_acc = 0
            start_time = time.time()
            for index, (image, targets) in tqdm(
                enumerate(test_loader), total=len(test_loader)
            ):
                image = image.to(device)
                targets = targets.to(device)
                
                outputs = model(image)
                pred = softmax(outputs, dim=1)
                total_epoch_acc += (pred.argmax(1) == targets).sum().item()
            
            end_time = time.time()
            test_time = end_time - start_time
        
        avg_epoch_acc = total_epoch_acc / len(test_mnist_dataset)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}],",
            f"Train Loss: {total_epoch_loss:.10f},",
            f"Used Time: {train_time * 1000:.3f}ms,",
            f"Test Acc: {avg_epoch_acc * 100:.3f}%,",
            f"Used Time: {test_time * 1000:.3f}ms",
        )
        train_loss.append(total_epoch_loss)
        test_acc.append(avg_epoch_acc * 100)
    return train_loss, test_acc