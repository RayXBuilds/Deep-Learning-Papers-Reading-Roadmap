# MobileNet v1：拆开空间滤波和通道混合，节省了哪部分计算？

[所属专题与原始资料](../README.md)

Andrew G. Howard、Menglong Zhu、Bo Chen、Dmitry Kalenichenko、Weijun Wang、Tobias Weyand、Marco Andreetto、Hartwig Adam。MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications，2017 年 arXiv v1 技术报告。[版本记录](https://arxiv.org/abs/1704.04861) · [原文](https://arxiv.org/html/1704.04861v1) · [TensorFlow-Slim 官方 v1 模型](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)

2026-11-21 核对。[预印本 DOI](https://doi.org/10.48550/arXiv.1704.04861) · [训练与视觉路线](../../../docs/roadmaps/training-and-vision.md)

## 问题

标准卷积为每一对输入、输出通道分别学一个空间核。移动端分类骨干希望减少一次推理的乘加和权重存储，还要能通过宽度与输入分辨率调整预算。

## 方法与关键推导

设输出有 $S$ 个空间位置，输入、输出宽度为 $M,N$，核边长为 $K$。标准卷积耗 $SK^2MN$ 次乘加。逐通道卷积只为每个输入通道做一次空间滤波，再用 1×1 卷积混合通道，总量为 $S(K^2M+MN)$，比值 $1/N+1/K^2$。宽度乘数同时缩放 M 和 N，所以逐点部分近似按其平方变化；分辨率乘数通过 S 影响两部分。

暂去 BN 和 ReLU，把输入通道 m 对各输出的有效核排成“输出通道 × 空间位置”矩阵，其元素是 $P_{nm}D_{muv}$。这是外积，秩至多为 1。因此任意训练好的标准卷积通常不能无损拆开。原网络两步间带有非线性，该线性推导只解释算子的约束。

## 原论文证据

第 4.1 节表 4 将完整卷积变体与 MobileNet 比较。ImageNet 准确率从 71.7% 到 70.6%，乘加从 4866M 到 569M，参数从 29.3M 到 4.2M。层级公式只数卷积乘加，表中数值对应整网。不能把乘加降幅直接换算为某一部手机上的延迟。

## 与其他论文的关系

[DenseNet](densenet.md) 用 1×1 卷积给后续空间卷积降宽，本篇的逐点卷积承担逐通道滤波之后的信息交换。作为分类骨干，MobileNet 可以提供 [SSD](../../06-detection-segmentation/notes/ssd.md) 所需的特征；换骨干和换检测头应分别比较。

## 可验证实验

已运行[线性卷积核分解对照](../../../experiments/results/vision-operators.md)。对同一输入分别计算两步算子与显式合成的稠密卷积，另给出两条线性无关的目标核，检验表达能力的边界。实验不含 BN、ReLU、训练或手机测速。

## 局限与思考

即使算术量少，内存搬运和算子实现也可能主导耗时。逐点层的占比还随通道宽度变化。可在固定设备上记录相同输入形状下的逐层预热后延迟，检查瓶颈是否随宽度乘数移动。
