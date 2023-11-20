import numpy as np
import torch
from utils import *


if __name__ == "__main__":
    learning_rate = 8e-2
    num_epochs = 101
    color = ["blue", "green", "orange", "purple"]
    for i in np.arange(4):
        weight_decay_rate = i / 4 * 0.01
        model = MNIST_CLS_Model(num_classes=10, dropout_rate=0)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)
        print(f"weight_decay_rate={weight_decay_rate}")
        train_loss, test_acc = train_MNIST_CLS(model, optimizer, num_epochs=num_epochs)