import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def my_one_hot(indices: torch.Tensor, num_classes: int):
    one_hot_tensor = torch.zeros(len(indices), num_classes).to(indices.device)
    one_hot_tensor.scatter_(1, indices.view(-1, 1), 1)
    return one_hot_tensor


class My_CrossEntropyLoss:
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor):
        max_values = torch.max(predictions, dim=1, keepdim=True).values
        exp_values = torch.exp(predictions - max_values)
        softmax_output = exp_values / torch.sum(exp_values, dim=1, keepdim=True)

        log_probs = torch.log(softmax_output)
        nll_loss = -torch.sum(targets * log_probs, dim=1)
        average_loss = torch.mean(nll_loss)
        return average_loss


class My_optimizer:
    def __init__(self, params: list[torch.Tensor], lr: float):
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            param.data = param.data - self.lr * param.grad.data

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.data.zero_()


class My_Linear:
    def __init__(self, input_feature: int, output_feature: int):
        self.weight = torch.randn(
            (output_feature, input_feature), requires_grad=True, dtype=torch.float32
        )
        self.bias = torch.randn(1, requires_grad=True, dtype=torch.float32)
        self.params = [self.weight, self.bias]

    def __call__(self, x: torch.Tensor):
        return self.forward(x)

    def forward(self, x: torch.Tensor):
        x = torch.matmul(x, self.weight.T) + self.bias
        return x

    def to(self, device: str):
        for param in self.params:
            param.data = param.data.to(device=device)
        return self

    def parameters(self):
        return self.params


class My_Flatten:
    def __call__(self, x: torch.Tensor):
        return self.forward(x)

    def forward(self, x: torch.Tensor):
        x = x.view(x.shape[0], -1)
        return x


class Model_3_1:
    def __init__(self, num_classes):
        self.flatten = My_Flatten()
        self.linear = My_Linear(28 * 28, num_classes)
        self.params = self.linear.params

    def __call__(self, x: torch.Tensor):
        return self.forward(x)

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.linear(x)
        return x

    def to(self, device: str):
        for param in self.params:
            param.data = param.data.to(device=device)
        return self

    def parameters(self):
        return self.params


learning_rate = 5e-3
num_epochs = 10
batch_size = 4096
num_classes = 10
device = "cuda:0" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)
train_dataset = datasets.FashionMNIST(
    root="./dataset", train=True, transform=transform, download=True
)
test_dataset = datasets.FashionMNIST(
    root="./dataset", train=False, transform=transform, download=True
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

model = Model_3_1(num_classes).to(device)
criterion = My_CrossEntropyLoss()
optimizer = My_optimizer(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    total_epoch_loss = 0
    for index, (images, targets) in tqdm(
        enumerate(train_loader), total=len(train_loader)
    ):
        optimizer.zero_grad()

        images = images.to(device)
        targets = targets.to(device).to(dtype=torch.long)

        one_hot_targets = (
            my_one_hot(targets, num_classes=num_classes).to(device).to(dtype=torch.long)
        )

        outputs = model(images)
        # ipdb.set_trace()
        loss = criterion(outputs, one_hot_targets)
        total_epoch_loss += loss

        loss.backward()
        optimizer.step()

    total_acc = 0
    with torch.no_grad():
        for index, (image, targets) in tqdm(
            enumerate(test_loader), total=len(test_loader)
        ):
            image = image.to(device)
            targets = targets.to(device)
            outputs = model(image)
            total_acc += (outputs.argmax(1) == targets).sum()
    print(
        f"Epoch {epoch + 1}/{num_epochs} Train, Loss: {total_epoch_loss}, Acc: {total_acc / len(test_dataset)}"
    )
