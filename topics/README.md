# 专题阅读总览

每个专题将中文精读、原始资料和待补环节放在同一入口。编号用于导航，不表示必须依次通读。

[首页](../README.md) · [逐篇索引](../docs/papers.md) · [年份索引](../docs/roadmaps/timeline.md) · [实验](../experiments/README.md)

| 专题 | 中文精读 | 原始 PDF | 阅读重点 |
|---|---:|---:|---|
| [基础与综述](01-foundations/README.md) | 0 | 4 | 从教材与综述建立术语，再进入训练、视觉和序列路线。 |
| [训练、归一化与优化](02-training/README.md) | 6 | 13 | 先读 Dropout、BN/LN 与 Adam，再比较 AdamW 和精确注意力计算。 |
| [视觉骨干与自监督](03-vision/README.md) | 12 | 4 | AlexNet → VGG / GoogLeNet → ResNet / DenseNet / MobileNet → ViT / Swin；自监督分支比较 MoCo、SimCLR、BYOL 与 MAE。 |
| [语言、序列与大模型](04-language/README.md) | 18 | 25 | word2vec → Seq2Seq → Attention → Transformer；再比较 ELMo、GPT、BERT、XLNet、T5，接续规模化、对齐与新架构。 |
| [生成模型与图像合成](05-generative/README.md) | 14 | 16 | VAE / GAN → WGAN / EBGAN / StyleGAN；DDPM → 潜空间扩散 → Flow Matching；艺术资料作为任务分支阅读。 |
| [目标检测与图像分割](06-detection-segmentation/README.md) | 4 | 9 | 先区分框与像素监督，再比较 Faster R-CNN、SSD、FPN 与 U-Net；返回视觉骨干理解特征来源。 |
| [图文理解与多模态](07-multimodal/README.md) | 2 | 3 | 从旧图像描述任务看视觉与语言如何连接，再对照 CLIP 的配对目标与 Janus-Pro 的理解/生成组织。 |
| [外部记忆与可微计算](08-memory/README.md) | 1 | 5 | 从记忆读写的问题出发，区分 Memory Networks、Neural Turing Machine 与 Pointer Networks。 |
| [强化学习与机器人](09-rl-robotics/README.md) | 1 | 17 | 先学习价值与策略优化，再进入控制、抓取、导航及仿真迁移；算法与应用分别归档。 |
| [迁移、少样本与参数适配](10-transfer/README.md) | 4 | 8 | 旧资料讨论蒸馏、持续学习和少样本；LoRA / QLoRA 是参数适配分支，阅读时区分目标。 |
| [视觉跟踪](11-tracking/README.md) | 1 | 3 | 先区分逐帧检测与跨帧目标关联，再阅读相关滤波、紧凑表示与全卷积跟踪资料。 |
| [框架实践与历史教程](12-practice/README.md) | 0 | 5 | 将旧框架教材和课件作为时代资料；动手验证以 experiments 的当前命令为准。 |

共 63 篇精读初版、112 个原始 PDF；两者不是同一计数口径。保留资料不等于完成精读，专题页分别标明覆盖与缺口。
