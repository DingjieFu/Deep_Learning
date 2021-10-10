# 用户：夜卜小魔王
# 线性回归模型的完整实现 未借助封装好的模块

import torch
import random
from d2l import torch as d2l


def synthetic_data(w, b, num_examples):
    # TODO:生成 y = Xw + b + 噪声
    x = torch.normal(0, 1, (num_examples, len(w)))  # lwn(w)列N个均值为0方差为1的数据
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape((-1, 1))


def data_iter(batch_size, features, labels):
    # TODO: 接收批量大小、特征矩阵和标签向量为输入， 生成大小为batch_size的小批量
    num_examples = len(features)  # 样本数量
    indices = list(range(num_examples))  # 对每个样本的index
    random.shuffle(indices)  # 打乱顺序 确保随机读取数据
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def linear_reg(X, w, b):
    # 线性回归模型
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    # 均⽅损失
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    # ⼩批量随机梯度下降 优化函数
    with torch.no_grad():  # 不用计算梯度
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()  # 清空梯度


true_w = torch.tensor([-2, -3.4])
true_b = 4.2
features_, labels_ = synthetic_data(true_w, true_b, 1000)  # 训练样本和标注
# print('features:', features_[0], '\nlabels:', labels_[0])  # 展示一组

# d2l.set_figsize((6, 6))
# d2l.plt.scatter(features_[:, 1].numpy(), labels_.numpy(), 5)  # 画出一列
# d2l.plt.xlabel("X"), d2l.plt.ylabel("Y"), d2l.plt.title("Linear-Regression Prediction")
# d2l.plt.show()

batch_size_ = 10
# for x, y in data_iter(batch_size, features, labels):
#     print(x, "\n", y)
#     break


# 定义初始化模型参数
w_ = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b_ = torch.zeros(1, requires_grad=True)

# 定义超参数
lr_ = 0.03  # 学习率
num_epochs = 5  # 循环批次
net = linear_reg  # 使用的模型
loss_ = squared_loss  # 定义的损失函数


for epoch in range(num_epochs):
    for X_, y_ in data_iter(batch_size_, features_, labels_):
        l_ = loss_(net(X_, w_, b_), y_)  # `X`和`y`的⼩批量损失
        # 因为`l`形状是(`batch_size`, 1)，⽽不是⼀个标量。`l`中的所有元素被加到⼀起，
        # 并以此计算关于[`w`, `b`]的梯度
        l_.sum().backward()
        sgd([w_, b_], lr_, batch_size_)  # 使⽤参数的梯度更新参数
    with torch.no_grad():
        train_l = loss_(net(features_, w_, b_), labels_)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w_.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b_}')
