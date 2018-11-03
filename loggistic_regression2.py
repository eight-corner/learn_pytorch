#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-11-3 下午1:14
# @Author  : YunSong
# @File    : loggistic_regression2.py
# @Software: PyCharm


import torch
import time
import numpy as np
from torch import nn
from torch.autograd import Variable


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.lr = nn.Linear(2, 1)
        self.sm = nn.Sigmoid()

    def forward(self, x):
        x = self.lr(x)
        x = self.sm(x)
        return x


if __name__ == '__main__':
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

    model = LogisticRegression()
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    start = time.time()
    for epoch in range(50000):
        # 前向传播
        x = Variable(x_data)
        y = Variable(y_data)
        out = model(x_data)
        loss = criterion(out, y)
        print_loss = loss.data[0]
        mask = out.ge(0.5).float()
        acc = (mask == y).sum().item() / x.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 1000 == 0:
            print('*' * 10)
            print('epoch {}'.format(epoch + 1))
            print('loss is : {:.4f}'.format(print_loss))
            print('acc is {:.4f}'.format(acc))

    during = time.time() - start
    print()
    print('During Time: {:.3f} s'.format(during))
