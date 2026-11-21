# 视觉骨干与自监督

[全部专题](../README.md) · [首页](../../README.md) · [年份索引](../../docs/roadmaps/timeline.md)

AlexNet → VGG / GoogLeNet → ResNet / DenseNet / MobileNet → ViT / Swin；自监督分支比较 MoCo、SimCLR、BYOL 与 MAE。

## 中文精读与阅读顺序

- [AlexNet：一张图像怎样变成一千个类别分数？](notes/alexnet.md)
- [VGG：把一个大卷积核拆成三层，保留了什么？](notes/vgg.md)
- [GoogLeNet：怎样让大核分支用得起？](notes/googlenet.md)
- [ResNet：为什么加深网络之后反而更难训练？](notes/resnet.md)
- [DenseNet：保留每层特征，为什么还能控制参数量？](notes/densenet.md)
- [MobileNet v1：拆开空间滤波和通道混合，节省了哪部分计算？](notes/mobilenet-v1.md)
- [ViT：图像块成为 token 后，模型还知道位置吗？](notes/vit.md)
- [Swin Transformer：移动半个窗口后，哪些图像块第一次能交流？](notes/swin.md)
- [MoCo：跨批负例还能用同一把尺子比较吗？](notes/moco.md)
- [SimCLR：为什么训练时用投影头，评估时读它的输入？](notes/simclr.md)
- [BYOL：没有负例时，在线网络在追谁？](notes/byol.md)
- [Masked Autoencoders（MAE）：解码器为什么要拿回原始块位置？](notes/mae.md)

## 原始资料

以下按历史文件名列出全部资料，文件字节保持不变。文件名可能缩写，不能据此推定版本或精读状态；同主题新笔记在上方，逐篇书目校订仍需核对原文。

- [Deep Residual Learning for Image Recognition](<papers/classification/Deep%20Residual%20Learning%20for%20Image%20Recognition.pdf>) · [同题精读](notes/resnet.md)
- [Going deeper with convolutions](<papers/classification/Going%20deeper%20with%20convolutions.pdf>) · [同题精读](notes/googlenet.md)
- [ImageNet Classification with Deep Convolutional](<papers/classification/ImageNet%20Classification%20with%20Deep%20Convolutional.pdf>) · [同题精读](notes/alexnet.md)
- [Very deep convolutional networks for large-scale image recognition](<papers/classification/Very%20deep%20convolutional%20networks%20for%20large-scale%20image%20recognition.pdf>) · [同题精读](notes/vgg.md)

## 状态与下一步

已有骨干及自监督精读；数据集训练状态见各篇。

[实验指南与实测边界](../../experiments/README.md) · [迁移对照](../../docs/archive/path-migration.md)
