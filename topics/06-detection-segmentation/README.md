# 目标检测与图像分割

[全部专题](../README.md) · [首页](../../README.md) · [年份索引](../../docs/roadmaps/timeline.md)

先区分框与像素监督，再比较 Faster R-CNN、SSD、FPN 与 U-Net；返回视觉骨干理解特征来源。

## 中文精读与阅读顺序

- [Faster R-CNN：如何让候选区域与检测器共用一张特征图？](notes/faster-rcnn.md)
- [SSD：多层预测怎样把候选框与分类合成一次前向传播？](notes/ssd.md)
- [FPN：同一幅图上，不同大小的目标应该读取哪一层？](notes/fpn.md)
- [U-Net：怎样用整块图像训练，又保住相邻细胞的边界？](notes/u-net.md)

## 原始资料

以下按历史文件名列出全部资料，文件字节保持不变。文件名可能缩写，不能据此推定版本或精读状态；同主题新笔记在上方，逐篇书目校订仍需核对原文。

- [Deep neural networks for object detection](<papers/detection/Deep%20neural%20networks%20for%20object%20detection.pdf>)
- [Proceedings of the IEEE International Conference on Computer Vision](<papers/detection/Proceedings%20of%20the%20IEEE%20International%20Conference%20on%20Computer%20Vision.pdf>)
- [Rich feature hierarchies for accurate object detection and semantic segmentation](<papers/detection/Rich%20feature%20hierarchies%20for%20accurate%20object%20detection%20and%20semantic%20segmentation.pdf>)
- [Unified, real-time object detection](<papers/detection/Unified%2C%20real-time%20object%20detection.pdf>)
- [DeepLab Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution,and Fully Connected CRFs](<papers/segmentation/DeepLab%20Semantic%20Image%20Segmentation%20with%20Deep%20Convolutional%20Nets%2C%20Atrous%20Convolution%2Cand%20Fully%20Connected%20CRFs.pdf>)
- [Fully Convolutional Networks for Semantic Segmentation](<papers/segmentation/Fully%20Convolutional%20Networks%20for%20Semantic%20Segmentation.pdf>)
- [Instance-aware Semantic Segmentation via Multi-task Network Cascades](<papers/segmentation/Instance-aware%20Semantic%20Segmentation%20via%20Multi-task%20Network%20Cascades.pdf>)
- [Instance-sensitive Fully Convolutional Networks](<papers/segmentation/Instance-sensitive%20Fully%20Convolutional%20Networks.pdf>)
- [Learning to Segment Object Candidates](<papers/segmentation/Learning%20to%20Segment%20Object%20Candidates.pdf>)

## 状态与下一步

R-CNN、YOLO、FCN、DeepLab 等旧资料仍待对应精读。

[实验指南与实测边界](../../experiments/README.md) · [迁移对照](../../docs/archive/path-migration.md)
