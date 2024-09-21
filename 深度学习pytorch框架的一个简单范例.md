---
title: 深度学习pytorch框架的一个简单范例
date: 2024-09-11 08:54:39
tags:
---


# 代码总览
```py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 准备数据
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 2. 定义一个简单的神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # 第一层，全连接层
        self.fc2 = nn.Linear(128, 64)     # 第二层，全连接层
        self.fc3 = nn.Linear(64, 10)      # 输出层，共 10 个类别

    def forward(self, x):
        x = x.view(-1, 28*28)  # 展平输入图像
        x = torch.relu(self.fc1(x))  # 第一层的激活函数
        x = torch.relu(self.fc2(x))  # 第二层的激活函数
        x = self.fc3(x)              # 输出层（不需要激活函数，因为会用交叉熵损失函数）
        return x

# 3. 创建模型实例，定义损失函数和优化器
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 训练模型
n_epochs = 5
for epoch in range(n_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()  # 清除之前的梯度
        outputs = model(images)  # 向前传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# 5. 测试模型
correct = 0
total = 0
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

```

# 头文件部分
头文件代码如下:
```py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
```
## torch库
torch库一般提供如下功能:  
张量运算 自动求导 神经网络模块 优化器 分布式计算 数据加载和处理 GPU支持

## torchvision库
1. torchvision.datasets 提供一些常用的数据集加载
2. torchvision.transforms 提供图像的预处理和数据增强
3. torchvision.models 提供一些预训练模型
4. torchvision.ops 常用的计算机视觉操作
5. torchvision.io 图像和视频的输入输出
6. torchvision.utils 可视化工具

# 数据准备部分
数据准备部分如下:
```py
# 1. 准备数据
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
```
## 数据预处理
通过transformer将图片进行预处理
```py
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
```
## 加载训练数据集
```py
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
```
## 加载测试数据集
```py
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
```
# 神经网络定义

```py
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # 第一层，全连接层
        self.fc2 = nn.Linear(128, 64)     # 第二层，全连接层
        self.fc3 = nn.Linear(64, 10)      # 输出层，共 10 个类别

    def forward(self, x):
        x = x.view(-1, 28*28)  # 展平输入图像
        x = torch.relu(self.fc1(x))  # 第一层的激活函数
        x = torch.relu(self.fc2(x))  # 第二层的激活函数
        x = self.fc3(x)              # 输出层（不需要激活函数，因为会用交叉熵损失函数）
        return x
```

1. 继承nn.model类
2. 覆盖__init__方法和forward方法
3. 神经网络的定义需要卸载init方法中，写在forward方法中可以运行，但是会导致内存泄漏。

# 创建模型实例，定义损失函数和优化器
```py
model = SimpleNN() #实例化模型
criterion = nn.CrossEntropyLoss() #交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

# 模型训练
```py
n_epochs = 5
for epoch in range(n_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()  # 清除之前的梯度
        outputs = model(images)  # 向前传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {running_loss/len(train_loader):.4f}")
```