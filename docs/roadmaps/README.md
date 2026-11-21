# 起步与核心路线

[首页](../../README.md) · [全部专题与原始资料](../../topics/README.md) · [论文索引](../papers.md) · [按年份阅读](timeline.md)

学习顺序不等于发表顺序。不要求先通读整本教材，也不要求把所有历史论文全部读完。

## 每篇论文的三遍阅读

1. 定位问题：看摘要、引言和主要图表，写出任务、输入输出、对比对象。
2. 还原方法：追踪一条输入经过模型的路径，标出张量形状、损失和训练/推理差异。
3. 检查证据：核对数据划分、基线、消融和指标，再运行机制实验。把未验证的推测留在问题清单。

采用 [精读模板](../../templates/paper-note.md)，用自己的话解释，避免只翻译摘要。

## 核心学习单元

时间由基础决定；下表是先修关系，不是强制周计划。

| 顺序 | 内容 | 验收 |
|---|---|---|
| 0 | 链式法则、均值方差、矩阵乘法、softmax | 手算两层函数导数与三元素 softmax |
| 1 | [ResNet](../../topics/03-vision/notes/resnet.md) | 区分训练退化与过拟合，比较普通/残差路径 |
| 2 | [BN](../../topics/02-training/notes/batch-normalization.md) → [LN](../../topics/02-training/notes/layer-normalization.md) | 在 3×3 矩阵上标出各自统计轴 |
| 3 | [Adam](../../topics/02-training/notes/adam.md) | 手算第一步偏差校正，比较至少两个学习率 |
| 4 | Encoder–Decoder → Attention → [Transformer](../../topics/04-language/notes/transformer.md) | 改变未来 token，确认掩码使早期输出不变 |
| 5（生成方向选读） | [VAE](../../topics/05-generative/notes/vae.md) / GAN 概念 → [DDPM](../../topics/05-generative/notes/ddpm.md) | 区分已知前向噪声、预测噪声与逆向采样 |

## 按目标分流

- [训练与视觉](training-and-vision.md)：训练机制、视觉架构。
- [序列与注意力](sequence-and-attention.md)：语言和多模态的基础组件。
- [生成模型](generative-models.md)：概率生成目标与加噪机制。
- 强化学习、机器人先查 [专题阅读入口](../../topics/09-rl-robotics/README.md)，尚未按新标准完成精读。

## 怎样判断读懂了

能回答：原方法卡在哪里？新方法改变哪一步？哪项实验支持它？在什么条件下可能不成立？自己运行的实验到底验证了哪一句话？

原论文数字与本地实验必须分开。公式单元测试不代表复现成功；下降的训练损失不代表泛化有效。
