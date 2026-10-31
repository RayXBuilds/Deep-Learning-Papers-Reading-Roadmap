# 论文索引与阅读状态

[首页](../README.md) · [路线](roadmaps/README.md) · [历史清单](archive/reading-list-2018.md)

“精读初版”表示已有中文问题分析、方法、原文证据位置、局限和机制实验；不表示完整复现。“原文入口”表示只完成选题和来源整理，精读与实验尚未完成。年份以首次预印本为主，必要时另列会议年份。此表是新路线的精选索引，不替代历史全部条目。

| 论文 / 年份 | 学习问题 | 一手来源 | 本仓库状态 |
|---|---|---|---|
| ResNet / 2015，CVPR 2016 | 深度增加后的训练退化 | [原文](https://arxiv.org/abs/1512.03385) | [精读初版](../notes/resnet.md)；residual |
| BatchNorm / 2015 | 跨样本统计与训练/推理 | [原文](https://arxiv.org/abs/1502.03167) | [精读初版](../notes/batch-normalization.md)；normalization |
| LayerNorm / 2016 | 单样本内部统计 | [原文](https://arxiv.org/abs/1607.06450) | [精读初版](../notes/layer-normalization.md)；normalization |
| Adam / 2014，ICLR 2015 | 自适应矩估计与偏差校正 | [原文](https://arxiv.org/abs/1412.6980) | [精读初版](../notes/adam.md)；adam |
| Bahdanau Attention / 2014，ICLR 2015 | 动态读取源序列 | [原文](https://arxiv.org/abs/1409.0473) | [精读初版](../notes/bahdanau-attention.md)；additive_attention.py |
| Transformer / 2017 | 用注意力组织序列计算 | [原文](https://arxiv.org/abs/1706.03762) | [精读初版](../notes/transformer.md)；attention |
| VAE / 2013，ICLR 2014 | 近似后验、变分目标与重参数化 | [原文](https://arxiv.org/abs/1312.6114) | [精读初版](../notes/vae.md)；vae_objectives.py |
| VAE 输入跳跃 / 2026 | 重构坐标与生成评价 | [原文](https://arxiv.org/abs/2610.37456) | [精读初版](../notes/vae-input-skip.md)；共享 VAE 机制实验；预印本评审状态未独立核实 |
| GAN / 2014 | 对抗式生成目标 | [原文](https://arxiv.org/abs/1406.2661) | 原文入口；精读待补 |
| DDPM / 2020 | 加噪与噪声预测 | [原文](https://arxiv.org/abs/2006.11239) | [精读初版](../notes/ddpm.md)；diffusion |
| AdamW / 2017，ICLR 2019 | 解耦权重衰减 | [原文](https://arxiv.org/abs/1711.05101) | [精读初版](../notes/adamw.md)；adamw_decay.py |
| BERT / 2018，NAACL 2019 | 双向 Transformer 预训练 | [原文](https://arxiv.org/abs/1810.04805) | [精读初版](../notes/bert.md)；复用 attention，MLM/NSP 未运行 |
| ViT / 2020，ICLR 2021 | 图像块序列的 Transformer | [原文](https://arxiv.org/abs/2010.11929) | 原文入口；精读待补 |
| CLIP / 2021 | 图文配对监督与对比学习 | [原文](https://arxiv.org/abs/2103.00020) | 原文入口；精读待补 |
| LoRA / 2021，ICLR 2022 | 冻结权重与低秩适配 | [原文](https://arxiv.org/abs/2106.09685) | 原文入口；精读待补 |

本表最近维护：2026-10-31；各论文的来源核对日期见精读元数据。新增内容优先引用公开一手来源，不新增论文 PDF 镜像。AlexNet、VGG、GoogLeNet 等历史条目见历史目录；其新格式精读尚未完成。
