# Dropout：什么量能在测试时用一次缩放估计？

Nitish Srivastava、Geoffrey Hinton、Alex Krizhevsky、Ilya Sutskever、Ruslan Salakhutdinov，*Dropout: A Simple Way to Prevent Neural Networks from Overfitting*，JMLR 15(56)，2014，1929–1958。核对于 2026-11-21。先修为 Bernoulli 分布、期望和非线性激活。作者页同时列出早期工作、Deepnet 代码和模型配置。[期刊原文](https://jmlr.org/papers/v15/srivastava14a.html) · [作者项目页](https://www.cs.toronto.edu/~nitish/dropout/)

## 问题

一个隐藏单元可以依赖另一个单元始终在场来纠正自己的输出。作者通过随机暂时移除单元，迫使同一套权重在不同的输入组合下工作。部署时若逐个计算所有子网，开销随可丢弃单元数指数增长。

## 方法与关键推导

按图 1、2 的原文约定，p 是保留概率。训练取独立的 mᵢ∼Bernoulli(p)，下一层激活前的值 z = b + Σ wᵢmᵢhᵢ。在固定当前 h 的条件下，E[z] = b + pΣwᵢhᵢ，Var(z) = p(1−p)Σ(wᵢhᵢ)²。因此在推理时缩放出边可精确匹配这一层的条件均值，偏置无需缩放。若在训练时使用 mᵢhᵢ/p，则推理直接用 hᵢ，这是另一套等价的均值约定，需要 p>0。

下一层若取 ReLU，正负响应跨越零点时，E[max(0,z)] 通常不同于 max(0,E[z])。这就是从线性期望走向整网预测近似时缺失的一步；多个子网还共享已训练的参数。

## 原论文证据

第 7 节图 10 固定 MNIST 网络结构，改变训练集大小，同时比较启用和关闭 dropout 的测试分类误差。样本数仅为 100、500 时作者没有观察到改善。图 11 另在推理端改变抽样子网数量，与权重缩放比较。前者考查泛化，后者考查怎样近似集成，两张图回答不同问题。

## 与其他论文的关系

[AlexNet](alexnet.md)的两个全连接隐藏层给出了实际使用位置。与 [AdamW](adamw.md) 对读时，可分别追踪随机掩码对每个样本的梯度和解耦衰减对每次参数更新的影响。

## 可验证实验

已枚举两个输入单元的四种掩码，比较仿射输出与 ReLU 输出的加权平均。输入、解析对照、通过条件及结果见[机制报告](../experiments/results/early-mechanisms.md)。

## 局限与思考

精确枚举使用固定权重，只检验推理近似。若把偏置调到所有子网响应均为正，ReLU 的近似差应消失。至于某个 p 能否改善测试误差，需要在独立验证集上选择 p 后重新评估。
