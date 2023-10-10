import torch

A = torch.tensor([[1, 2, 3]])

B = torch.tensor([[4],
                  [5]])

# 方法1: 使用PyTorch的减法操作符
result1 = A - B

# 方法2: 使用PyTorch的sub函数
result2 = torch.sub(A, B)

# 方法3: 手动实现广播机制并作差
def mysub(a:torch.Tensor, b:torch.Tensor):
    if not (
        (a.size(0) == 1 and b.size(1) == 1) 
        or 
        (a.size(1) == 1 and b.size(0) == 1)
        ):
        raise ValueError("输入的张量大小无法满足广播机制的条件。")
    else:
        target_shape = torch.Size([max(A.size(0), B.size(0)), max(A.size(1), B.size(1))])
        A_broadcasted = A.expand(target_shape)
        B_broadcasted = B.expand(target_shape)
        result = torch.zeros(target_shape, dtype=torch.int64).to(device=A_broadcasted.device)
        for i in range(target_shape[0]):
            for j in range(target_shape[1]):
                result[i, j] = A_broadcasted[i, j] - B_broadcasted[i, j]
        return result

result3 = mysub(A, B)

print("方法1的结果:")
print(result1)
print("方法2的结果:")
print(result2)
print("方法3的结果:")
print(result3)
