#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/3 11:24
# @Author  : PanYunSong
# @File    : poly_model.py
# @Software: PyCharm

import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable

def make_features(x):
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, 4)], 1)

res = make_features(torch.Tensor([3]))

W_target = torch.Tensor([0.5, 3, 2.4]).unsqueeze(1)
b_target = torch.Tensor([0.9])

def f(x):
    return x.mm(W_target) + b_target[0]


def get_batch(batch_size=32):
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    if torch.cuda.is_available():
        return Variable(x).cuda(), Variable(y).cuda()
    return Variable(x), Variable(y)


class poly_model(nn.Module):
    def __init__(self):
        super(poly_model, self).__init__()
        self.poly = nn.Linear(3, 1)

    def forward(self, x):
        out = self.poly(x)
        return out


if torch.cuda.is_available():
    model = poly_model().cuda()
else:
    model = poly_model()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

epoch = 0
while True:
    batch_x, batch_y = get_batch()
    output = model(batch_x)
    loss = criterion(output, batch_y)
    print_loss = loss.data[0]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch += 1

    if (epoch + 1) % 20 == 0:
        print('Epoch[{}],loss:{:.6f} '.format(epoch + 1, loss.data[0]))

    if print_loss < 1e-3:
        break
