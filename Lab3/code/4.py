import torch
from torch import nn
from utils import *
from torch.utils.data import random_split


learning_rate = 1e-3
num_epochs = 161
batch_size = 8192
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

train_dataset_length = int(0.8 * len(train_mnist_dataset))
val_dataset_length = len(train_mnist_dataset) - train_dataset_length
train_mnist_dataset, val_mnist_dataset = random_split(
    train_mnist_dataset,
    [train_dataset_length, val_dataset_length],
    generator=torch.Generator().manual_seed(42),
)

train_loader = DataLoader(dataset=train_mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=14, pin_memory=True)
val_loader = DataLoader(dataset=val_mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=14, pin_memory=True)
test_loader = DataLoader(dataset=test_mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=14, pin_memory=True)

model = MNIST_CLS_Model(num_classes=10, dropout_rate=0.2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

early_stopping_patience = 5
best_val_loss = float("inf")
current_patience = 0

train_loss = list()
test_acc = list()
val_loss = list()
for epoch in range(num_epochs):
    model.train()
    total_epoch_loss = 0
    for index, (images, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()

        images = images.to(device)
        targets = targets.to(device)
        one_hot_targets = one_hot(targets, num_classes=num_classes).to(dtype=torch.float)

        outputs = model(images)
        loss = criterion(outputs, one_hot_targets)
        total_epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        total_epoch_acc = 0
        for index, (image, targets) in tqdm(enumerate(test_loader), total=len(test_loader)):
            image = image.to(device)
            targets = targets.to(device)

            outputs = model(image)
            pred = softmax(outputs, dim=1)
            total_epoch_acc += (pred.argmax(1) == targets).sum().item()
        avg_epoch_acc = total_epoch_acc / len(test_mnist_dataset)

        val_total_epoch_loss = 0
        for index, (image, targets) in tqdm(enumerate(val_loader), total=len(test_loader)):
            image = image.to(device)
            targets = targets.to(device)
            one_hot_targets = one_hot(targets, num_classes=num_classes).to(dtype=torch.float)

            outputs = model(image)
            loss = criterion(outputs, one_hot_targets)
            val_total_epoch_loss += loss.item()

    print(
        f"Epoch [{epoch + 1}/{num_epochs}],",
        f"Train Loss: {total_epoch_loss:.10f},",
        f"Test Acc: {avg_epoch_acc * 100:.3f}%,",
        f"Val Loss: {val_total_epoch_loss:.10f}",
    )
    train_loss.append(total_epoch_loss)
    test_acc.append(avg_epoch_acc * 100)
    val_loss.append(val_total_epoch_loss)

    if val_total_epoch_loss < best_val_loss:
        best_val_loss = val_total_epoch_loss
        current_patience = 0
    else:
        current_patience += 1
        if current_patience >= early_stopping_patience:
            print(f"Early stopping after {epoch + 1} epochs.")
            break
