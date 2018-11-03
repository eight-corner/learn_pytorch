#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-11-3 下午12:21
# @Author  : YunSong
# @File    : logistic_regression.py
# @Software: PyCharm

import torch
import time
import numpy as np
from torch import nn
from torch.autograd import Variable

torch.manual_seed(2017)

# 从 data.txt 中读入点
with open('./data/data.txt', 'r') as f:
    data_list = [i.split('\n')[0].split(',') for i in f.readlines()]
    data = [(float(i[0]), float(i[1]), float(i[2])) for i in data_list]

# 标准化
x0_max = max([i[0] for i in data])
x1_max = max([i[1] for i in data])
data = [(i[0] / x0_max, i[1] / x1_max, i[2]) for i in data]

np_data = np.array(data, dtype='float32')  # 转换成 numpy array
x_data = torch.from_numpy(np_data[:, 0:2])  # 转换成 Tensor, 大小是 [100, 2]
y_data = torch.from_numpy(np_data[:, -1]).unsqueeze(1)  # 转换成 Tensor，大小是 [100, 1]

x_data = Variable(x_data)
y_data = Variable(y_data)

w = nn.Parameter(torch.randn(2, 1))
b = nn.Parameter(torch.zeros(1))


def logistic_reg(x):
    return torch.mm(x, w) + b


criterion = nn.BCEWithLogitsLoss()  # 将 sigmoid 和 loss 写在一层，有更快的速度、更好的稳定性

y_pred = logistic_reg(x_data)
loss = criterion(y_pred, y_data)
optimizer = torch.optim.SGD([w, b], lr=1.)

start = time.time()
for e in range(1000):
    # 前向传播
    y_pred = logistic_reg(x_data)
    loss = criterion(y_pred, y_data)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 计算正确率
    mask = y_pred.ge(0.5).float()
    acc = (mask == y_data).sum().item() / y_data.shape[0]
    if (e + 1) % 100 == 0:
        print('epoch: {}, Loss: {:.5f}, Acc: {:.5f}'.format(e + 1, loss.item(), acc))

during = time.time() - start
print()
print('During Time: {:.3f} s'.format(during))
