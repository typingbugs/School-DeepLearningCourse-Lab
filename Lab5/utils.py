import math
import torch
from torch.utils import data
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import time


def mse_fn(y, pred):
    return np.mean((np.array(y) - np.array(pred)) ** 2)


def mae_fn(y, pred):
    return np.mean(np.abs(np.array(y) - np.array(pred)))


def mape_fn(y, pred):
    mask = y != 0
    y = y[mask]
    pred = pred[mask]
    mape = np.abs((y - pred) / y)
    mape = np.mean(mape) * 100
    return mape


def eval(y, pred):
    y = y.cpu().numpy()
    pred = pred.cpu().numpy()
    mse = mse_fn(y, pred)
    rmse = math.sqrt(mse)
    mae = mae_fn(y, pred)
    mape = mape_fn(y, pred)
    return [rmse, mae, mape]


# 测试函数（用于分类）
def test(net, data_iter, loss_fn, denormalize_fn, device='cpu'):
    rmse, mae, mape = 0, 0, 0
    batch_count = 0
    total_loss = 0.0
    net.eval()
    for seqs, targets in data_iter:
        seqs = seqs.to(device).float()
        targets = targets.to(device).float()
        y_hat = net(seqs)
        loss = loss_fn(y_hat, targets)

        targets = denormalize_fn(targets)
        y_hat = denormalize_fn(y_hat)
        a, b, c = eval(targets.detach(), y_hat.detach())
        rmse += a
        mae += b
        mape += c
        total_loss += loss.detach().cpu().numpy().tolist()
        batch_count += 1
    return [rmse / batch_count, mae / batch_count, mape / batch_count], total_loss / batch_count


def train(net, train_iter, val_iter, test_iter, loss_fn, denormalize_fn, optimizer, num_epoch,
          early_stop=10, device='cpu', num_print_epoch_round=0):
    train_loss_lst = []
    val_loss_lst = []
    train_score_lst = []
    val_score_lst = []
    epoch_time = []

    best_epoch = 0
    best_val_rmse = 9999
    early_stop_flag = 0
    for epoch in range(num_epoch):
        net.train()
        epoch_loss = 0
        batch_count = 0
        batch_time = []
        rmse, mae, mape = 0, 0, 0
        for seqs, targets in train_iter:
            batch_s = time.time()
            seqs = seqs.to(device).float()
            targets = targets.to(device).float()
            optimizer.zero_grad()
            y_hat = net(seqs)
            loss = loss_fn(y_hat, targets)
            loss.backward()
            optimizer.step()

            targets = denormalize_fn(targets)
            y_hat = denormalize_fn(y_hat)
            a, b, c = eval(targets.detach(), y_hat.detach())
            rmse += a
            mae += b
            mape += c
            epoch_loss += loss.detach().cpu().numpy().tolist()
            batch_count += 1

            batch_time.append(time.time() - batch_s)

        train_loss = epoch_loss / batch_count
        train_loss_lst.append(train_loss)
        train_score_lst.append([rmse/batch_count, mae/batch_count, mape/batch_count])

        # 验证集
        val_score, val_loss = test(net, val_iter, loss_fn, denormalize_fn, device)
        val_score_lst.append(val_score)
        val_loss_lst.append(val_loss)

        epoch_time.append(np.array(batch_time).sum())

        # 打印本轮训练结果
        if num_print_epoch_round > 0 and (epoch+1) % num_print_epoch_round == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epoch}],",
                f"Train Loss: {train_loss:.4f},",
                f"Train RMSE: {train_score_lst[-1][0]:.4f},",
                f"Val Loss: {val_loss:.4f},",
                f"Val RMSE: {val_score[0]:.6f},",
                f"Time Use: {epoch_time[-1]:.3f}s"
            )

        # 早停
        if val_score[0] < best_val_rmse:
            best_val_rmse = val_score[0]
            best_epoch = epoch
            early_stop_flag = 0
        else:
            early_stop_flag += 1
            if early_stop_flag == early_stop:
                print(f'The model has not been improved for {early_stop} rounds. Stop early!')
                break

    # 输出最终训练结果
    print(
        f'Final result:',
        f'Get best validation rmse {np.array(val_score_lst)[:, 0].min():.4f} at epoch {best_epoch},',
        f'Total time {np.array(epoch_time).sum():.2f}s'
    )

    # 计算测试集效果
    test_score, test_loss = test(net, test_iter, loss_fn, denormalize_fn, device)
    print(
        'Test result:',
        f'Test RMSE: {test_score[0]},',
        f'Test MAE: {test_score[1]},',
        f'Test MAPE: {test_score[2]}'
    )
    return train_loss_lst, val_loss_lst, train_score_lst, val_score_lst, epoch


def visualize(num_epochs, train_data, test_data, x_label='epoch', y_label='loss'):
    x = np.arange(0, num_epochs + 1).astype(dtype=np.int32)
    plt.figure(figsize=(5, 3.5))
    plt.plot(x, train_data, label=f"train_{y_label}", linewidth=1.5)
    plt.plot(x, test_data, label=f"val_{y_label}", linewidth=1.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def plot_metric(score_log):
    score_log = np.array(score_log)

    plt.figure(figsize=(13, 3.5))
    plt.subplot(1, 3, 1)
    plt.plot(score_log[:, 0], c='#d28ad4')
    plt.ylabel('RMSE')

    plt.subplot(1, 3, 2)
    plt.plot(score_log[:, 1], c='#e765eb')
    plt.ylabel('MAE')

    plt.subplot(1, 3, 3)
    plt.plot(score_log[:, 2], c='#6b016d')
    plt.ylabel('MAPE')

    plt.show()