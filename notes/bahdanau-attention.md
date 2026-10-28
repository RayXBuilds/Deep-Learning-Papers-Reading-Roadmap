# Bahdanau Attention：每一步解码，应该重新读取哪里？

[首页](../README.md) · [序列路线](../docs/roadmaps/sequence-and-attention.md) · [实验报告](../experiments/results/additive-attention.md)

- 原题：Neural Machine Translation by Jointly Learning to Align and Translate。
- 作者：Dzmitry Bahdanau、Kyunghyun Cho、Yoshua Bengio。
- 年份/版本：2014-09-01 首次预印本；ICLR 2015 oral；本次核对 arXiv v7（2016-05-19）。[元数据](https://arxiv.org/abs/1409.0473) · [作者页面](https://yoshuabengio.org/en/publication/neural-machine-translation-jointly-learning-align-and-translate)。
- 原文位置：[第 3.1–3.2 节、式 (5)–(6)、附录 A.1.2/A.2.2；表 1、图 2–3](https://arxiv.org/html/1409.0473v7)。
- 作者实现：[GroundHog](https://github.com/lisa-groundhog/GroundHog)，由论文第 4 节脚注给出；仓库已声明停止开发，依赖 Theano，本次未安装或运行。
- 先修：Encoder–Decoder、RNN 状态、矩阵乘法、softmax；后续阅读 [Transformer](transformer.md)。
- 收录理由：补齐固定表示与 Transformer 之间的动态读取环节。本论文已在旧清单中，本次补充中文精读与实验，不重复计为新论文。
- 状态：精读初版、确定性机制实验已验证；没有训练翻译模型或复现 BLEU。

## 问题

如果所有解码步骤只能使用同一个上下文向量，读取源端信息的方式就不随当前需求变化。该论文让解码器保留访问源端各位置表示的能力。

每一步得到的上下文仍然是固定维度向量；改变的是“能否反复访问整组源端表示”，不是把每个向量变成任意长度。

## 方法与关键推导

论文使用双向 RNN 生成源端 annotation h_j；打分网络结合上一解码状态 s_(i−1) 与 h_j：

```text
e_ij = vᵀ tanh(Wq s_(i−1) + Wh h_j)
α_ij = exp(e_ij) / Σ_k exp(e_ik)
c_i  = Σ_j α_ij h_j
```

符号 Wq、Wh 是本笔记对原文 Wa、Ua 的重命名。若 query 维度为 d_q、annotation 维度为 d_h、隐藏打分维度为 d_a，则 Wq 为 d_a×d_q、Wh 为 d_a×d_h、v 长度为 d_a，最终 c_i 长度为 d_h。query 与 annotation 无须同维。

**本项目的代数练习：非线性为什么有意义？** 如果直接删除 tanh，则分数可拆为只依赖 query 的常数与只依赖 h_j 的项。对所有源位置共享的 query 常数会在 softmax 中抵消，因此这个线性可分形式的权重不随 query 改变。下面的实验专门加入此消融；这不是声称任何线性注意力都无效。

变长组批中的 padding 不是有效源词。教学实现先限定 softmax 的有效位置，再补零权重。这里是源端 padding 掩码，不是目标端的因果掩码；离线翻译可以读取完整源句。

## 原论文证据

表 1 比较 RNNencdec 与 RNNsearch；图 2 按源句长度分组；图 3 展示软对齐。核对时注意 All 与 No UNK 不是同一测试子集，带星号模型训练更久。原模型也包含双向编码器等设计差异，不能把完整系统结果当成只修改一个算子的严格消融。

本仓库不引用这些 BLEU 数值作为本地成绩；以下数值全部来自人工构造的输入。

## 与其他论文的关系

- Encoder–Decoder → 本文：把每步固定上下文改成动态读取。
- 本文 → [Transformer](transformer.md)：提供 query 决定读取权重的概念前置；两者的打分函数、循环结构和多头设计不同。
- 本实验 ↔ 已有 attention 实验：前者检查源序列 padding，后者检查目标序列未来位置不可见，不能混用两类掩码。

## 可验证实验

```bash
python3 experiments/additive_attention.py --output /tmp/additive-attention.json
python3 -m unittest discover -s experiments -p 'test_*.py' -v
```

使用三个二维源表示、两个一维 query 和手工投影。依次比较动态读取、固定均值、去掉 tanh、增加两项极端 padding。另将有效长度改为 2，与截断后的短序列对照。

验收：动态上下文最大变化大于 1；固定均值和线性可分打分的 query 变化小于 1e-12；屏蔽 padding 后长短序列上下文变化均小于 1e-12；去掉屏蔽的对照变化大于 1。形状、单有效位置、全掩码输入拒绝和手算 softmax 都有独立测试。

本次动态上下文变化为 2.095444，padding 后变化为 0。完整配置与结果见 [实测报告](../experiments/results/additive-attention.md)。固定均值只是教学对照，不是重新实现论文中的 RNNencdec 基线。

## 局限与思考

投影是手工选择的，没有学习语义、翻译损失、词表或数据集。机制验证只能说明此计算图具有预期性质，不能证明自动学会对齐或改善翻译。

尝试只改变有效位置的 annotation，并预测上下文变化；再把它移入 padding 区。解释为什么屏蔽位置的值即使很大也不应参与求和。若 query 根本没有改变权重，应先检查非线性和参数，再讨论是否“没有学会注意力”。
