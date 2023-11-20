import torch
from utils import *

if __name__ == "__main__":
    learning_rate = 5e-2
    num_epochs = 161
    color = ["blue", "green", "orange"]
    optim_names = ["SGD", "RMSprop", "Adam"]

    model = MNIST_CLS_Model(num_classes=10, dropout_rate=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    print(f"optimizer: SGD")
    train_loss, test_acc = train_MNIST_CLS(model, optimizer, num_epochs=num_epochs)

    model = MNIST_CLS_Model(num_classes=10, dropout_rate=0)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99, eps=1e-8)
    print(f"optimizer: RMSprop")
    train_loss, test_acc = train_MNIST_CLS(model, optimizer, num_epochs=num_epochs)

    model = MNIST_CLS_Model(num_classes=10, dropout_rate=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    print(f"optimizer: Adam")
    train_loss, test_acc = train_MNIST_CLS(model, optimizer, num_epochs=num_epochs)
