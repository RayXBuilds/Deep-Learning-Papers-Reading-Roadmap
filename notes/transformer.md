# Transformer：Q、K、V 与信息可见性

[首页](../README.md) · [序列路线](../docs/roadmaps/sequence-and-attention.md) · [实验说明](../experiments/README.md)

- 原文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)，Vaswani 等，2017。
- 核对位置：第 3.1–3.5 节、第 4 节、表 1 与表 3。
- 先修：矩阵乘法、softmax、[LN](layer-normalization.md)、Encoder–Decoder。
- 状态：中文精读初版；单头机制实验；未复现机器翻译。

## 问题

序列模型需要在不同位置之间传递信息。Transformer 用注意力组织这种交互，并结合前馈网络、位置表示、残差与归一化构成编码器和解码器；它不只是一个注意力公式。

## 方法与关键推导

Attention(Q,K,V)=softmax(QKᵀ/sqrt(d_k))V。Q 为 n×d_k、K 为 m×d_k、V 为 m×d_v；分数为 n×m，沿每个 query 对应的 key 轴归一化，输出 n×d_v。

在独立、零均值、单位方差分量的假设下，点积方差随 d_k 增长，缩放用于控制尺度。因果掩码把不可见未来位置的分数设为负无穷，再做 softmax，不能等输出算完才把某些结果清零。

## 原论文证据

表 1 是机器翻译结果，表 3 比较结构变化。训练时可并行计算多个位置，不代表自回归生成时能一次知道所有未来 token。标准全注意力的分数矩阵随序列长度平方增长，不能只看到并行收益。

## 与其他论文的关系

Bahdanau Attention 为动态读取源序列提供前置理解；Transformer 的自注意力重新组织序列计算。残差与 LN 是共享组件。BERT、ViT 分别改变预训练任务与输入表示，具体关系见 [关联表](../docs/connections.md)。

## 可验证实验

```bash
python3 experiments/run.py --experiment attention
```

固定三个位置的 Q、K、V，比较有/无因果掩码。把最后位置的 K、V 大幅改变：因果版本前两个输出应不变，无掩码版本应发生变化。另用全零 Q 验证等权平均，作为可手算参照。

## 局限与思考

这里没有多头投影、位置编码、词嵌入或训练，不能生成文本。思考：没有位置表示时，自注意力如何对待排列？cross-attention 的 Q 与 K、V 为什么可以来自不同序列？
