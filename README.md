# Deep Learning Papers Reading Roadmap

**深度学习经典论文精读与技术演进路线**

从经典论文出发，理解深度学习中的关键问题、方法选择与技术演进。每篇中文精读围绕“为什么提出、怎样实现、证据是什么、局限在哪里”展开，并配套可在 CPU 上运行的小实验。

**[开始阅读](docs/roadmaps/README.md) · [论文索引](docs/papers.md) · [技术关联](docs/connections.md) · [实验指南](experiments/README.md) · [历史资料](docs/archive/README.md)**

适合已经接触过 Python 和神经网络、希望系统阅读原论文的学习者。建议具备矩阵运算、链式法则和基础概率知识；具体先修要求见各条路线。

## 从哪里开始

1. **刚开始读论文**：先看 [阅读方法与核心顺序](docs/roadmaps/README.md)，再从 [ResNet：为什么加深网络之后反而更难训练？](notes/resnet.md) 开始。
2. **已有研究方向**：从下面选择路线，结合原论文与中文精读阅读；需要补前置概念时，再返回相关笔记。
3. **想验证自己的理解**：运行配套实验，比较预测与结果，再回到论文检查解释是否成立。

## 选择阅读路线

| 方向 | 围绕什么问题阅读 | 路线入口 |
|---|---|---|
| 训练与视觉 | 网络为什么能加深？归一化与优化器各自改变什么？ | [卷积网络 → 残差连接 → 训练方法 → ViT](docs/roadmaps/training-and-vision.md) |
| 序列与注意力 | 模型如何读取上下文？注意力怎样组织序列计算？ | [Encoder–Decoder → Attention → Transformer](docs/roadmaps/sequence-and-attention.md) |
| 生成模型 | 模型如何学习数据分布？不同生成目标怎样比较？ | [VAE / GAN → DDPM](docs/roadmaps/generative-models.md) |

路线包含已完成精读和后续阅读入口，具体状态见 [论文索引](docs/papers.md)。箭头表示建议阅读顺序；方法之间的继承、组件联系与对照关系见 [技术关联](docs/connections.md)。

## 已有中文精读

| 论文 | 这篇精读回答什么 | 配套实验 |
|---|---|---|
| [ResNet](notes/resnet.md) | 为什么增加恒等捷径会改变深层网络的梯度传播？ | 普通路径与残差路径的导数对照 |
| [Batch Normalization](notes/batch-normalization.md) | 归一化统计量来自哪里，为什么会受同批样本影响？ | 改变同批样本，观察输出变化 |
| [Layer Normalization](notes/layer-normalization.md) | 改变统计轴后，哪些批次依赖消失了？ | 与 BatchNorm 使用相同输入对照 |
| [Adam](notes/adam.md) | 梯度历史如何影响参数更新，为什么需要偏差校正？ | 二次函数优化与手算第一步更新 |
| [Bahdanau Attention](notes/bahdanau-attention.md) | 每一步解码如何重新读取源端信息，padding 怎样排除？ | 加性打分、非线性消融与变长掩码 |
| [Transformer](notes/transformer.md) | Q、K、V 怎样协作，因果掩码如何限制信息可见性？ | 修改未来位置，检查前缀输出 |
| [VAE](notes/vae.md) | 近似后验、下界和重参数化怎样配合？ | 高斯闭式参照与采样梯度 |
| [VAE 输入跳跃](notes/vae-input-skip.md) | 重构误差下降为何不保证生成分布更好？ | 与 VAE 共用实验：密度、坐标缩放与负对照 |
| [DDPM](notes/ddpm.md) | 加噪过程怎样产生监督信号，预测噪声意味着什么？ | 高斯加噪与独立测试集上的线性去噪 |

目前有 **9 篇精读初版、7 个机制实验**。笔记中提供原文链接、关键公式、证据位置与延伸问题；实验用于理解局部机制，完整论文基准复现尚未完成。

## 动手验证

已有 Python 3.10+ 即可运行，无需 GPU、第三方依赖或外部数据。在仓库根目录执行：

```bash
python3 experiments/run.py --experiment all --seed 7
```

上面的命令运行核心五项实验。加性注意力专题单独运行：`python3 experiments/additive_attention.py`；VAE 目标与评价专题运行：`python3 experiments/vae_objectives.py`。

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
