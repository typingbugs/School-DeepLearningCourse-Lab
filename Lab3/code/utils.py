import torch
from torch.nn.functional import *
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets, transforms
from tqdm import tqdm

import ipdb


class MNIST_CLS_Model(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=28 * 28, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_MNIST_CLS(model, optimizer, num_epochs):
    batch_size = 8192
    num_classes = 10
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_mnist_dataset = datasets.MNIST(
        root="../dataset", train=True, transform=transform, download=True
    )
    test_mnist_dataset = datasets.MNIST(
        root="../dataset", train=False, transform=transform, download=True
    )
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

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    train_loss = list()
    test_acc = list()
    for epoch in range(num_epochs):
        model.train()
        total_epoch_loss = 0
        for index, (images, targets) in tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):
            optimizer.zero_grad()

            images = images.to(device)
            targets = targets.to(device)
            one_hot_targets = one_hot(targets, num_classes=num_classes).to(
                dtype=torch.float
            )

            outputs = model(images)
            loss = criterion(outputs, one_hot_targets)
            total_epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            total_epoch_acc = 0
            for index, (image, targets) in tqdm(
                enumerate(test_loader), total=len(test_loader)
            ):
                image = image.to(device)
                targets = targets.to(device)

                outputs = model(image)
                pred = softmax(outputs, dim=1)
                total_epoch_acc += (pred.argmax(1) == targets).sum().item()

        avg_epoch_acc = total_epoch_acc / len(test_mnist_dataset)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}],",
            f"Train Loss: {total_epoch_loss:.10f},",
            f"Test Acc: {avg_epoch_acc * 100:.3f}%,",
        )
        train_loss.append(total_epoch_loss)
        test_acc.append(avg_epoch_acc * 100)
    return train_loss, test_acc
