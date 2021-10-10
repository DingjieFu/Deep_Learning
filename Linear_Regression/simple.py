# 用户：夜卜小魔王
# 通过封装好的模块

import torch
import torch.nn as nn
from torch.utils import data
from d2l import torch as d2l


def load_array(data_arrays, batch_size, is_train=True):
    # 构造⼀个PyTorch数据迭代器
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

batch_size_ = 10
data_iter = load_array((features, labels), batch_size_)
# print(next(iter(data_iter)))

net = nn.Sequential(nn.Linear(2, 1))  # 线性网络
net[0].weight.data.normal_(0, 0.01)  # 权重
net[0].bias.data.fill_(0)  # 偏置

loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l_ = loss(net(X), y)
        trainer.zero_grad()
        l_.backward()
        trainer.step()
    l_ = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l_:f}')
