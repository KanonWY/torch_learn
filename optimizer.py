# PyTorch 中的 优化器（optimizer）用于根据计算出的梯度更新模型的参数，使模型更接近于目标。
# 优化器主要通过 梯度下降 方法来调整参数.
# 常用的优化算法包括 SGD（随机梯度下降）、Adam、RMSProp 等。

# 核心功能
# （1）更新参数
# （2）选择学习率
# （3）使用不同的优化算法

# 常见优化器
# （1）SGD 随机梯度下降
# （2）Adam 自适应矩估计

# 学习率
# 在优化器中，lr 是 学习率（Learning Rate），
# 它是一个超参数，控制着每次参数更新时的步长（即每次更新的幅度）。
# 学习率在优化过程中非常重要，直接影响训练的效果和速度。
# 学习率的作用：
# （1）小学习率：更新的步伐较小，训练过程较为稳定，但可能需要更多的训练步骤才能收敛，训练时间较长。
# （2）大学习率：更新的步伐较大，可能加速收敛，但也容易导致训练过程中的震荡，甚至错过最优解，或者不收敛。
# 学习率的调整
# （1）固定学习率
# （2）动态学习率

import torch.optim as optim
import torch
from torch import nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = SimpleModel()
optimizer = optim.SGD(model.parameters(),lr = 0.01)
