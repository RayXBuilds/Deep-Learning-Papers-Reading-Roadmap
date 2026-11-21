# 训练与视觉

[路线入口](README.md) · [论文索引与原文](../papers.md)

AlexNet → VGG / GoogLeNet（并列设计路线）→ ResNet → ViT（架构对照）。BN、LN、Adam 是穿插阅读的训练组件。

| 阶段 | 核心问题 | 状态与验收 |
|---|---|---|
| AlexNet | 卷积、非线性、正则化和计算资源怎样组成系统？ | [AlexNet](../../notes/alexnet.md) 与 [Dropout](../../notes/dropout.md)；画出输入输出，区分结构与训练技巧 |
| VGG / GoogLeNet | 小卷积堆叠与多分支分别权衡什么？ | [VGG](../../notes/vgg.md) 与 [GoogLeNet](../../notes/googlenet.md)；比较感受野、参数量和计算量 |
| ResNet | 加深后训练误差恶化，能否改变参数化？ | [精读](../../notes/resnet.md)；解释输入导数不是准确率 |
| BN / LN | 统计对象和推理方式怎样不同？ | [BN](../../notes/batch-normalization.md)、[LN](../../notes/layer-normalization.md)；改变同批其他样本；用 bn_jacobian.py 检查统计量求导与冻结对照 |
| Adam | 梯度历史怎样影响步长？ | [精读](../../notes/adam.md)；记录不同学习率与停止条件 |
| AdamW | L2 正则进入矩估计后，和参数直接衰减有何区别？ | [精读](../../notes/adamw.md)；核对首步、矩状态与学习率约定 |
| ViT | 图像块成为 token 后，先验和数据需求怎样变化？ | [精读](../../notes/vit.md)；块投影/卷积等价、位置排列实验；分类训练未复现 |

跨模态延伸：[CLIP 精读](../../notes/clip.md)。在 ViT/ResNet 编码器之外，单独分析图文配对监督与文本类别接口；固定向量实验不代表分类准确率。

## 下一阶段实验（尚未实现）

同一图像数据集、相同深度和宽度，对比普通网络与残差网络。固定划分、增强、训练预算，运行至少三个种子；记录训练损失、验证指标、参数量和耗时。若同时更换结构与学习率，不能把差异全部归因于残差。

当前标量导数实验不满足上述完整复现条件。

2026-11-19 增加 [ResNet 激活与抵消补充](../../notes/resnet.md)：运行 `python3 experiments/residual_gate.py`，对照 D(I+J_F) 与仅有 I+J_F 的差别。[实测](../../experiments/results/residual-gate.md)。数据集训练计划仍未完成。

## 从分类骨干走向空间预测

| 任务 | 阅读顺序 | 自查问题 |
|---|---|---|
| 分类骨干 | [ResNet](../../notes/resnet.md) → [DenseNet](../../notes/densenet.md)，再读 [MobileNet v1](../../notes/mobilenet-v1.md) | 相加或拼接之后还能否单独读取旧通道？逐通道滤波之后哪一步交换通道信息？ |
| 检测 | [Faster R-CNN](../../notes/faster-rcnn.md) → [SSD](../../notes/ssd.md)（检测头对照）→ [FPN](../../notes/fpn.md)（多尺度特征） | 分别指出候选框、类别预测和尺度分配发生的位置。 |
| 像素分割 | [U-Net](../../notes/u-net.md)，再与 FPN 图 3 对读 | 编码特征为什么要裁剪？哪一级输出像素类别，哪几级提供区域特征？ |

## 视图、窗口与遮挡

先给一张图做两次增强，分别画出 [SimCLR](../../notes/simclr.md) 的批内候选、[MoCo](../../notes/moco.md) 的队列和 [BYOL](../../notes/byol.md) 的回归目标，再把正例换成配对文本，比较 [CLIP](../../notes/clip.md) 的候选矩阵。

沿 [ViT](../../notes/vit.md) 读两种计算预算：[Swin](../../notes/swin.md) 限制每个查询的邻居并交替分窗，[MAE](../../notes/mae.md) 预训练时从编码器输入中移除被遮块。画出两者在同一张图上保留的位置与连接。
