# Deep Learning Papers Reading Roadmap

**深度学习经典论文精读与技术演进路线**

从经典论文出发，理解深度学习中的关键问题、方法选择与技术演进。每篇中文精读围绕“为什么提出、怎样实现、证据是什么、局限在哪里”展开，部分论文配有可在 CPU 上运行的小实验。

**[开始阅读](docs/roadmaps/README.md) · [论文索引](docs/papers.md) · [技术关联](docs/connections.md) · [实验指南](experiments/README.md) · [历史资料](docs/archive/README.md)**

适合已经接触过 Python 和神经网络、希望系统阅读原论文的学习者。建议具备矩阵运算、链式法则和基础概率知识；具体先修要求见各条路线。

## 从哪里开始

1. **刚开始读论文**：先看 [阅读方法与核心顺序](docs/roadmaps/README.md)，再从 [ResNet：为什么加深网络之后反而更难训练？](notes/resnet.md) 开始。
2. **已有研究方向**：从下面选择路线，结合原论文与中文精读阅读；需要补前置概念时，再返回相关笔记。
3. **想验证自己的理解**：运行配套实验，比较预测与结果，再回到论文检查解释是否成立。

## 选择阅读路线

| 方向 | 围绕什么问题阅读 | 路线入口 |
|---|---|---|
| 训练与视觉 | 网络怎样加深或轻量化？检测和分割怎样利用空间特征？ | [分类骨干、训练方法、目标检测与像素分割](docs/roadmaps/training-and-vision.md) |
| 序列与注意力 | 模型如何读取上下文？注意力怎样组织序列计算？ | [Encoder–Decoder → Attention → Transformer](docs/roadmaps/sequence-and-attention.md) |
| 生成模型 | 模型如何学习数据分布？不同生成目标怎样比较？ | [VAE / GAN → DDPM → Flow Matching](docs/roadmaps/generative-models.md) |

路线包含已完成精读和后续阅读入口，具体状态见 [论文索引](docs/papers.md)。箭头表示建议阅读顺序；方法之间的继承、组件联系与对照关系见 [技术关联](docs/connections.md)。

## 已有中文精读

目前有 **58 篇精读初版、32 个机制实验专题**。从 [AlexNet](notes/alexnet.md) 开始可对照小核堆叠、多尺度分支与残差连接。从 [word2vec](notes/word2vec.md) 和 [Seq2Seq](notes/seq2seq.md) 开始可追踪词向量、固定句向量与注意力之间的联系。

沿[视觉路线](docs/roadmaps/training-and-vision.md)可从批内对比、动量字典读到无负样本回归和掩码重建。沿[序列路线](docs/roadmaps/sequence-and-attention.md)可比较上下文特征、因果预测、排列预测与片段去噪，再读指令微调与偏好优化；沿[生成路线](docs/roadmaps/generative-models.md)可从 GAN 读到 StyleGAN 的分层风格控制，也可从 DDPM 读到潜空间扩散。

近年的两个读点是 [Mamba-2 的状态递推](notes/mamba-2.md)与 [LLaDA 的离散去噪](notes/llada.md)，均可运行 CPU 对照。

[按年份浏览全部精读](docs/roadmaps/timeline.md) · [逐篇查看原文和验证状态](docs/papers.md)

## 动手验证

已有 Python 3.10+ 即可运行，无需 GPU、第三方依赖或外部数据。在仓库根目录执行：

```bash
python3 experiments/run.py --experiment all --seed 7
```

上面的命令运行核心五项实验。其余专题的命令和验收条件见[实验指南](experiments/README.md)。

也可以把 `all` 替换为 `residual`、`normalization`、`adam`、`attention` 或 `diffusion`，单独运行感兴趣的实验。

[查看输入、对照与验收条件](experiments/README.md) · [查看实测结果](experiments/results/README.md)

## 经典资料与其他方向

原有资料涵盖深度学习基础、语音识别、强化学习、迁移学习、单样本学习、目标检测、跟踪、分割、图像描述、机器翻译、机器人及风格迁移。

- [按专题查找资料](docs/archive/README.md)：保留 22 个原分类目录和 112 个 PDF 的路径。
- [查看历史论文清单](docs/archive/reading-list-2018.md)：主体整理于 2018 年，阅读时注意版本与时代背景。
- [查看已核实的勘误](docs/errata.md)：包括归一化、跟踪方法和书目信息的修正。

## 参与维护

欢迎补充中文精读、技术关联、实验与纠错。优先完成一个学习问题的解释和验证；新增论文采用作者、会议或 arXiv 的一手来源链接。

[贡献规范](CONTRIBUTING.md) · [后续选题](docs/maintenance.md) · [文档与目录导航](docs/README.md)

历史论文与书籍的版权归各自权利人。
