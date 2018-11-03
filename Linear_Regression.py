#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/3 11:04
# @Author  : PanYunSong
# @File    : Linear_Regression.py
# @Software: PyCharm
# @Version : python3

import sys

import torch
import numpy as np
from torch.autograd import Variable
from torch import nn, optim

# 读入数据 x 和 y
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

x_train = torch.from_numpy(x_train)

y_train = torch.from_numpy(y_train)


class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out


if torch.cuda.is_available():
    model = LinearRegression().cuda()
else:
    model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

num_epochs = 1000
for epoch in range(num_epochs):
    inputs = Variable(x_train)
    target = Variable(y_train)
    out = model(inputs)
    loss = criterion(out, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        print('Epoch[{}/{}],loss:{:.6f} '.format(epoch + 1, num_epochs, loss.data[0]))

model.eval()
model.cpu()
predict = model(Variable(x_train))
predict = predict.data.numpy()

import matplotlib.pyplot as plt

plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
plt.plot(x_train.numpy(), predict, label='Fitting Line')
plt.show()
