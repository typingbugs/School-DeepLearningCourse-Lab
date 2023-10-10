import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import ipdb


class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(28 * 28, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.linear(x)
        return x


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

model = Model(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    total_epoch_loss = 0
    model.train()
    for index, (images, targets) in tqdm(
        enumerate(train_loader), total=len(train_loader)
    ):
        optimizer.zero_grad()

        images = images.to(device)
        targets = targets.to(device)

        one_hot_targets = (
            torch.nn.functional.one_hot(targets, num_classes=num_classes)
            .to(device)
            .to(dtype=torch.float32)
        )

        outputs = model(images)
        loss = criterion(outputs, one_hot_targets)
        total_epoch_loss += loss

        loss.backward()
        optimizer.step()

    model.eval()
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
