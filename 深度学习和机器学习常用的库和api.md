# pytorch api
## pyorch的数据类型
```
torch.float32 (或 torch.float)
torch.float64 (或 torch.double)
torch.float16
torch.int32 (或 torch.int)
torch.int64 (或 torch.long)
torch.uint8
torch.bool
torch.complex64
torch.complex128
```

## tensor的初始化
```py
import torch
# 空矩阵
x = torch.empty(5, 3)
# 随机矩阵
x = torch.rand(5, 3)
# 零矩阵
x = torch.zeros(5, 3, dtype=torch.long)
# 初始化为固定值
x = torch.tensor([5.5, 3])

x = x.new_ones(5, 3, dtype=torch.float64) # 返回的tensor默认具有相同的  torch.dtype和torch.device

x = torch.randn_like(x, dtype=torch.float) # 指定新的数据类型

```
## 矩阵形状转换
```py
import torch

# 创建一个张量
x = torch.randn(2, 3, 4)  # Shape: (2, 3, 4)

# 1. reshape: 重塑张量形状
reshaped = x.reshape(6, 4)  # Shape: (6, 4)

# 2. view: 类似于 reshape，但要求张量是连续的
viewed = x.view(6, 4)  # Shape: (6, 4)

# 3. permute: 改变维度顺序
permuted = x.permute(1, 0, 2)  # Shape: (3, 2, 4)

# 4. transpose: 交换两个指定的维度
transposed = x.transpose(1, 2)  # Shape: (2, 4, 3)

# 5. unsqueeze: 在指定位置插入一个维度
unsqueezed = torch.unsqueeze(x, 1)  # Shape: (2, 1, 3, 4)

# 6. squeeze: 删除指定位置的大小为1的维度
squeezed = torch.squeeze(unsqueezed, 1)  # Shape: (2, 3, 4)

# 7. flatten: 将指定范围内的维度展平成一维
flattened = torch.flatten(x, start_dim=1)  # Shape: (2, 12)

# 8. cat: 沿指定维度拼接张量
y = torch.randn(2, 3, 4)
concatenated = torch.cat([x, y], dim=0)  # Shape: (4, 3, 4)

# 9. stack: 在新维度上叠加张量
stacked = torch.stack([x, y], dim=0)  # Shape: (2, 2, 3, 4)

# 10. repeat: 沿指定维度重复张量
repeated = x.repeat(2, 1, 1)  # Shape: (4, 3, 4)

# 11. tile: 沿指定维度复制张量（类似 repeat）
tiled = x.tile((2, 1, 1))  # Shape: (4, 3, 4)

# 打印所有结果
print("Original shape:", x.shape)
print("Reshaped shape:", reshaped.shape)
print("Viewed shape:", viewed.shape)
print("Permuted shape:", permuted.shape)
print("Transposed shape:", transposed.shape)
print("Unsqueezed shape:", unsqueezed.shape)
print("Squeezed shape:", squeezed.shape)
print("Flattened shape:", flattened.shape)
print("Concatenated shape:", concatenated.shape)
print("Stacked shape:", stacked.shape)
print("Repeated shape:", repeated.shape)
print("Tiled shape:", tiled.shape)
```
## 矩阵的存储空间问题

## 卷积，线性变换，池化api
```py
# 线性变换默认对张量的最后一个维度进行变换

# 输入张量 (batch_size=2, time_steps=3, sequence_length=4, in_features=5)
x = torch.randn(2, 3, 4, 5)

# 定义线性层
linear = nn.Linear(in_features=5, out_features=6)

# 输出张量
output = linear(x)  # Shape: (2, 3, 4, 6)
```

```py
# 卷积时，大小不足会进行广播，但是输入维度过大会报错
# 例如1d卷积，输入必须是[B,C,X]. 2d卷积，输入必须是[B,C,H,W]
torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

```

