## 模型的模式
在 PyTorch 中，模型的模式指的是模型在不同阶段（如训练、验证、测试）下的行为模式。根据不同的阶段，某些层或机制（如 Dropout 和 BatchNorm）会表现得不同。因此，PyTorch 提供了两种常用的模式来切换模型的行为：
- train mode
- eval mode
### 优化方式与参数
#### Dropout
Dropout 是一种 正则化 技术，用于减少神经网络中的过拟合。它通过在每次训练过程中随机丢弃一部分神经元（即设置为零），强制网络不会依赖于某一特定神经元，从而提高模型的泛化能力。
工作原理：
- 在每个训练步骤中，Dropout 会以一定的概率将神经元的输出设置为零（即“丢弃”该神经元）。
- 丢弃的概率由超参数 p 控制（例如 p=0.5 表示每个神经元有 50% 的概率会被丢弃）。
- 在推理阶段（即模型评估阶段），所有神经元都被使用，但为了保持训练时的期望输出，通常会将神经元的输出按丢弃概率进行缩放。
作用：
- 防止过拟合：通过在训练过程中丢弃神经元，模型不会过度依赖某些特定的神经元，从而减少过拟合的风险。
- 提高泛化能力：模型学会使用不同的神经元组合，从而使得训练过程更加鲁棒
```python
import torch
import torch.nn as nn
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10,20)
        # 50% 的概率丢弃神经元
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(20,2)

    def forward(self, x):
        x = self.fc1(x)
        # 训练时，50% 的神经元会被丢弃
        x = self.self.dropout(x)
        x = self.fc2(x)
        return x
```
#### Batch Normalization
BatchNorm 是一种用于加速训练并提高神经网络稳定性的技术。它通过对每一层的输入进行标准化（即使其均值为 0，方差为 1）来避免梯度消失或爆炸问题，并且能够加速模型的收敛。  
工作原理：
- 标准化：对每个 mini-batch 中的输入特征进行标准化，使得每个输入的均值为 0，方差为 1。
- 缩放和平移：在标准化之后，BatchNorm 会通过学习的 缩放系数（γ） 和 平移系数（β） 来恢复原始分布的尺度和位置
作用：
- 加速训练：标准化后的数据使得梯度传播更平稳，有助于加速网络的收敛。
- 缓解梯度消失问题：通过标准化输入特征，避免了深层网络中梯度消失或梯度爆炸的问题。
- 提高泛化能力：BatchNorm 在一定程度上具有正则化效果，有助于提高模型的泛化能力。
```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.batchnorm = nn.BatchNorm1d(20)  # 对 20 维输入进行 BatchNorm
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.batchnorm(x)  # 进行 BatchNorm 处理
        x = self.fc2(x)
        return x
```
### 模式的区别
在 PyTorch 中，模型的模式指的是模型在不同阶段（如训练、验证、测试）下的行为模式。根据不同的阶段，某些层或机制（如 Dropout 和 BatchNorm）会表现得不同。因此，PyTorch 提供了两种常用的模式来切换模型的行为：
#### 训练模式
在训练模式下，模型的某些层（如 Dropout 和 BatchNorm）会根据当前批次的数据进行特定的操作。
- Dropout：会随机丢弃神经元，避免过拟合。
- BatchNorm：会根据当前 batch 的均值和方差进行标准化。

#### 评估模式
在评估模式下，模型的某些行为会发生变化:
- Dropout：会关闭（即所有神经元都参与计算），确保每个神经元都参与到推理中，从而保证模型在评估时稳定性。
- BatchNorm：会使用训练阶段学习到的全局均值和方差来进行标准化，而不是当前 batch 的均值和方差。这保证了在推理时一致性。