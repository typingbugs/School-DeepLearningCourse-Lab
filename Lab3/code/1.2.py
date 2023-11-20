import numpy as np
import torch
from utils import *


if __name__ == "__main__":
    learning_rate = 8e-2
    num_epochs = 101
    for i in np.arange(3):
        dropout_rate = 0.1 + 0.4 * i
        model = MNIST_CLS_Model(num_classes=10, dropout_rate=dropout_rate)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        print(f"dropout_rate={dropout_rate}")
        train_loss, test_acc = train_MNIST_CLS(model, optimizer, num_epochs=num_epochs)
