import torch
import torch.nn as nn

y_pred = torch.tensor([2.5, 0.0, 2.1], dtype=torch.float32)
y_true = torch.tensor([3.0, -0.5, 2.0], dtype=torch.float32)


def mse_learn():
    # 均方误差
    # 对异常值敏感，因为平方会放大误差
    mse_loss = nn.MSELoss()

    loss = mse_loss(y_pred, y_true)
    print(loss.item())


def mae_learn():
    # 平均绝对误差
    l1_loss = nn.L1Loss()
    loss = l1_loss(y_pred, y_true)
    print(loss.item())


def mse_and_mae_learn():
    # 当误差较小时，使用 MSE；误差较大时，使用 MAE，避免异常值影
    smooth_l1_loss = nn.SmoothL1Loss()
    loss = smooth_l1_loss(y_pred, y_true)
    print(loss.item())


mse_learn()
mae_learn()
mse_and_mae_learn()


def cross_entropy_loss_learn():
    # 交叉熵
    # y_pred 不需要经过 Softmax，因为 CrossEntropyLoss 内部会自动进行 log_softmax
    # y_true 需要是类别索引（而不是独热编码）
    cross_entropy_loss = nn.CrossEntropyLoss()
    y_pred = torch.tensor([[2.0, 1.0, 0.1]], dtype=torch.float32)  # 预测 logits
    y_true = torch.tensor([0])  # 真实类别索引
    loss = cross_entropy_loss(y_pred, y_true)
    print(loss.item())  # 输出损失值


def bce_loss_learn():
    # y_pred 必须是 Sigmoid 之后的概率（范围在 0~1 之间）
    # y_true 取值 0 或 1

    bce_loss = nn.BCELoss()
    y_pred = torch.tensor([0.9, 0.2, 0.3])  # 预测概率
    y_true = torch.tensor([1.0, 0.0, 0.0])  # 真实标签

    loss = bce_loss(y_pred, y_true)
    print(loss.item())


cross_entropy_loss_learn()
bce_loss_learn()
