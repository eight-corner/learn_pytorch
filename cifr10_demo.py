#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/4 10:12
# @Author  : PanYunSong
# @File    : cifr10_demo.py
# @Software: PyCharm


import torchvision as tv
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision.transforms as transforms

# 定义对数据的预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
])

# 训练集
trainset = tv.datasets.CIFAR10(
    root='./data/',
    train=True,
    download=False,
    transform=transform)

trainloader = t.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=2)

# 测试集
testset = tv.datasets.CIFAR10(
    './data/',
    train=False,
    download=False,
    transform=transform)

testloader = t.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=False,
    num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    net = Net()
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    t.set_num_threads(5)
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # 更新参数
            optimizer.step()
            # 打印log信息
            # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
            running_loss += loss.item()
            if i % 2000 == 1999:  # 每2000个batch打印一下训练状态
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')

    dataiter = iter(testloader)
    images, labels = dataiter.next()  # 一个batch返回4张图片
    print('实际的label: ', ' '.join('%08s' % classes[labels[j]] for j in range(4)))

    # 计算图片在每个类别上的分数
    outputs = net(images)
    # 得分最高的那个类
    _, predicted = t.max(outputs.data, 1)
    print('预测结果: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    correct = 0  # 预测正确的图片数
    total = 0  # 总共的图片数
    # 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
    with t.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = t.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

    print('10000张测试集中的准确率为: %d %%' % (100 * correct / total))
