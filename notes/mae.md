# Masked Autoencoders（MAE）：解码器为什么要拿回原始块位置？

Masked Autoencoders Are Scalable Vision Learners

- 作者：Kaiming He、Xinlei Chen、Saining Xie、Yanghao Li、Piotr Dollár、Ross Girshick。
- 版本：2021-11（v1），2021-12（v3），CVPR 2022。核对于 2026-11-21。[预印本与版本记录](https://arxiv.org/abs/2111.06377) · [会议原文](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html) · [作者仓库](https://github.com/facebookresearch/mae)
- 先修：[ViT](vit.md)、均方误差与排列的逆。
- 预印本 DOI：10.48550/arXiv.2111.06377。
- 状态：精读初版，位置还原与掩码误差已验证，完整基准未复现。

## 问题

相邻像素高度相关，少量遮挡容易靠局部插值完成。大量随机丢块提高了从上下文恢复内容的难度。若遮住的块仍用占位符送入昂贵编码器，计算开销却几乎原样保留。

## 方法与关键推导

将 n 个位置随机打乱，只把前 k 个块及其原位置编码送入 ViT。解码时用较窄的投影映射可见表示，补齐 n−k 个可学习占位符，逆置换后加入解码器的位置编码。没有位置信息时，同一个占位符无法指明应重建天空还是地面。

设遮挡集合为 M，块内有 d 个像素分量，目标为 $L=\sum_{i\in M}\|\hat x_i-x_i\|^2/(|M|d)$。原文还比较逐块归一化的像素目标。若遮 75%，编码器注意分数矩阵约缩至 1/16；线性层、解码器、访存仍有成本，整机加速不能按十六倍估计。

## 原论文证据

图 5 在 ImageNet-1K 上改变随机遮挡率，冻结线性探针比端到端微调更敏感。表 1c 在 ViT-L/16 的对照中，把 mask token 移出编码器后线性探针从 59.6% 到 73.5%，微调从 84.2% 到 84.9%。读表时需保留评估方式，以及默认实验使用未归一化像素这一条件。

## 与其他论文的关系

[ViT](vit.md) 的块序列允许先挑出可见位置；[BERT](bert.md) 仍在完整输入位置上计算上下文，预测离散词表标签。与 [BYOL](byol.md) 对照，MAE 的监督由原图给出，目标不随网络更新。

## 可验证实验

见[掩码重建实验](../experiments/results/masked-patches.md)。改变可见位置预测，对比仅遮挡区和全图 MSE；用位置编号核对逆置换，避免损失索引和可视化图像一起错位。

## 局限与思考

低重建误差可能源于局部纹理预测。固定遮挡比例，改为整块连续遮挡，再比较像素误差和线性探针的排序是否一致；这需要重新训练与独立下游评估。
