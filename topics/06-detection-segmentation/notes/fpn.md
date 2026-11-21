# FPN：同一幅图上，不同大小的目标应该读取哪一层？

[所属专题与原始资料](../README.md)

Tsung-Yi Lin、Piotr Dollár、Ross Girshick、Kaiming He、Bharath Hariharan、Serge Belongie。Feature Pyramid Networks for Object Detection，2016 年预印，CVPR 2017；采用 2017 年 v2。[版本记录](https://arxiv.org/abs/1612.03144) · [原文](https://arxiv.org/html/1612.03144v2) · [作者团队后续发布的 Detectron FPN 实现](https://github.com/facebookresearch/Detectron/blob/main/detectron/modeling/FPN.py)

2026-11-21 核对。[CVPR 会议版](https://doi.org/10.1109/CVPR.2017.106) · [训练与视觉路线](../../../docs/roadmaps/training-and-vision.md)

## 问题

深层特征能区分类别，却可能把小物体压缩到单个格点。浅层保留位置，却少了深层上下文。逐个尺度缩放图像并运行骨干开销很大；FPN 要用一次骨干前向计算构造可供多个检测尺度读取的表示。

## 方法与关键推导

从 ResNet 各阶段末端取 $C_2$ 至 $C_5$，步幅分别为 4、8、16、32。顶层做 1×1 投影，递归令 $M_l=\mathrm{proj}(C_l)+\mathrm{up}_2(M_{l+1})$。两项统一为 256 通道后逐元素相加；各层再对 $M_l$ 做 3×3 卷积得到检测用 $P_l$。上采样传递的是合并后的中间图，输出卷积缓和上采样伪影。

RPN 在 P2–P6 各层用单一面积、多种长宽比的锚框生成候选，P6 由 P5 再下采样得到。Fast R-CNN 为每个 RoI 选择一层池化。式 (1) 对图像坐标下的宽高取 $\lfloor4+\log_2(\sqrt{wh}/224)\rfloor$，限制到 P2–P5；边长翻倍，选层升一级。固定面积改变长宽比时路由不变，故一个极细长框和等面积正方形会读相同步幅的网格。

## 原论文证据

表 2 固定 COCO minival、ResNet-50 与来自 FPN-RPN 的候选，比对 Fast R-CNN 头所用特征。完整金字塔 AP 为 33.9，去横向连接为 31.3，仅自底向上为 24.9。三者均用 2fc 头，这组差值有助于定位融合路径的贡献。表 1 单独报告候选召回，不能将其中 AR 当成最终 AP。

## 与其他论文的关系

[ResNet](../../03-vision/notes/resnet.md) 提供不同步幅的阶段输出，[Faster R-CNN](faster-rcnn.md) 提供 RPN 和区域分类器，FPN 改的是两者之间的特征供给。[SSD](ssd.md) 也多层预测，却没有这条自顶向下融合路径。与 [U-Net](u-net.md) 的区别还在于各尺度的监督位置。

## 可验证实验

已运行[尺度路由对照](../../../experiments/results/vision-operators.md)。固定同一组框，对比只用 P4 和按面积分层后的特征格点跨度，含边界截断和等面积细长框。未运行 RPN、融合卷积或 COCO 评测。

## 局限与思考

分层让中等尺度区间的几何平均跨度较稳定，但超出层级范围时会饱和，细长框的短边仍可能不足一个格点。可在相同面积桶内按短边长度拆分 AP，区分面积路由与采样分辨率各自的限制。
