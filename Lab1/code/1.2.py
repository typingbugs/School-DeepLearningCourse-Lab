import torch

mean = 0
stddev = 0.01

P = torch.normal(mean=mean, std=stddev, size=(3, 2))
Q = torch.normal(mean=mean, std=stddev, size=(4, 2))

print("矩阵 P:")
print(P)
print("矩阵 Q:")
print(Q)

# 对矩阵Q进行转置操作，得到矩阵Q的转置Q^T
QT = Q.T
print("矩阵 QT:")
print(QT)

# 计算矩阵P和矩阵Q^T的矩阵相乘
result = torch.matmul(P, QT)
print("矩阵相乘的结果:")
print(result)

