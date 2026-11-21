# Chinchilla：同一笔训练预算该买参数还是 token？

[所属专题与原始资料](../README.md)

- 原文：[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)。
- 作者与版本：Jordan Hoffmann、Sebastian Borgeaud、Arthur Mensch 等，2022 年 arXiv v1。NeurIPS 2022 的索引题名为 An empirical analysis of compute-optimal large language model training（同一研究）。[会议记录及 DOI](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c1e2faff6f588870935f114ebe04a3e5-Abstract.html)；[作者说明](https://deepmind.google/blog/an-empirical-analysis-of-compute-optimal-large-language-model-training/)未提供完整预训练代码。
- 书目核对（2026-11-21）：[会议 DOI 10.52202/068431-2176](https://doi.org/10.52202/068431-2176)。
- 先修：[Transformer](transformer.md)、幂律拟合与对数导数。
- 状态：精读初版。

## 问题

给定 FLOPs 上限 C，提高参数量 N 会挤占能处理的训练 token 数 D。把不同模型各自训练很久再比较，会丢失这个约束。作者要找的是同一预算下终点语言建模损失最低的组合。

## 方法与关键推导

对于稠密 Transformer，取 C≈6ND。令损失模型 L=E+A/N^α+B/D^β，代入 D=C/(6N)，对 log N 求导得 αA/N^α=βB/D^β。左侧是增加容量的边际收益，右侧是少读数据的边际代价。因此 N 随 C 的幂指数为 β/(α+β)，D 的指数为 α/(α+β)。固定每个参数对应多少 token，只有两个指数恰好相等时才由这个模型推出。

## 原论文证据

§3 表 2 用三条路径交叉核对。沿训练曲线取最低损失的包络，得到参数指数 0.50；固定 FLOPs 扫模型大小得到 0.49；拟合全部终点得到 0.46。§4 用约 1.4 万亿 token 训练 70B Chinchilla，与预算相近的 280B Gopher 比较。该大规模对照同时改变了训练设置，单一指标差不能归因于某个指数。

## 与其他论文的关系

[Transformer](transformer.md) 给出每层算子，本工作在该类稠密模型上分配整个训练预算。[FlashAttention](../../02-training/notes/flashattention.md) 改变相同算术的访存成本，因而等 FLOPs 的两次训练仍可有不同耗时。

## 可验证实验

本批未运行训练。可在相同语料和分词器下选三个模型规模，分别把训练长度设为 C/(6N)，使学习率退火在各自终点结束，再比较独立验证集的每 token 交叉熵。至少三个预算点才适合讨论最优规模如何移动。

## 局限与思考

损失模型忽略了数据重复、质量及推理次数。若部署后生成 token 远多于训练 token，同一训练前沿上较小的模型可能节省更多总成本；应另外绘制训练加推理成本曲线。
