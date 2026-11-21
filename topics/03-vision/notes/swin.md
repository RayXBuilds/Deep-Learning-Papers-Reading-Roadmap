# Swin Transformer：移动半个窗口后，哪些图像块第一次能交流？

[所属专题与原始资料](../README.md)

Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows

- 作者：Ze Liu、Yutong Lin、Yue Cao、Han Hu、Yixuan Wei、Zheng Zhang、Stephen Lin、Baining Guo。
- 版本：2021-03（v1），2021-08（v2），ICCV 2021。核对于 2026-11-21。[预印本与版本记录](https://arxiv.org/abs/2103.14030) · [会议原文](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html) · [作者仓库](https://github.com/microsoft/Swin-Transformer)
- 先修：[ViT](vit.md)、二维网格与注意力掩码。
- 预印本 DOI：10.48550/arXiv.2103.14030。
- 状态：精读初版，本地机制实验未运行，完整基准未复现。

## 问题

密集预测要保留较细的空间特征，也要整合远处信息。像素边长翻倍，固定块尺寸下 token 数变为四倍，全局相似度矩阵随之变为十六倍。固定非重叠窗口能省算力，但反复在原窗计算会留下互不通信的区域。

## 方法与关键推导

每个 M×M 窗内独立注意。若有 n 个 token，相似度元素共 $(n/M^2)(M^2)^2=nM^2$，全局基线为 $n^2$；此计数未包括线性投影。连续两层先用常规窗，再平移约半窗。第二层的一次聚合便能读取前层不同窗的输出。循环位移便于批量计算，边界掩码禁止图像首尾因 roll 而误成邻居。

图 3 的 patch merging 拼接相邻 2×2 位置，通道从 4C 投影到 2C，空间位置数降至四分之一。若每行有四个块、窗宽为二，常规窗先聚合 {0,1} 和 {2,3}；第二层的移窗 {1,2} 已能间接看到四个输入，边界处的 0 和 3 则不应直接相连。窗口移动改变连接，多尺度合并改变特征分辨率，两者应分别消融。

## 原论文证据

第 4.4 节表 4 固定 Swin-T 和任务评估协议，对比全程固定分窗与交替移窗。移窗带来 ImageNet-1K top-1 增加 1.1 个百分点、COCO box AP 增加 2.8、ADE20K mIoU 增加 2.8；各指标对应不同下游头，不能互相换算。

## 与其他论文的关系

[ViT](vit.md) 提供块 token 与注意力主干；Swin 将单尺度全局交互换成分层局部交互。[MAE](mae.md) 也缩减注意力计算，但减少的是预训练编码器实际收到的位置数。

## 可验证实验

未运行 Swin 网络。可用二维网格的布尔邻接矩阵相乘，比较两层固定窗与一次移窗的可达集合；单独检查上下边与左右边，定位误用循环边界的实现。

## 局限与思考

连通性只说明存在路径，未给出路径实际权重。可在相同 FLOPs 下测试固定窗、移窗的实测延迟，再按物体大小分层比较检测误差。
