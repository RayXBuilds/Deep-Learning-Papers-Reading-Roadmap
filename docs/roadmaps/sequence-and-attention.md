# 序列与注意力

[路线入口](README.md) · [论文索引与原文](../papers.md)

1. Encoder–Decoder：画出输入表示与逐步解码。原始序列模型资料在历史目录。
2. [Bahdanau Attention](../../notes/bahdanau-attention.md)：比较固定上下文与动态读取，运行加性打分、非线性消融和变长 padding 实验。
3. Transformer：完成 [精读](../../notes/transformer.md) 和 attention 实验，区分 self-attention、cross-attention、因果掩码。
4. [BERT 精读](../../notes/bert.md)：区分双向可见性、输入扰动与损失位置；已有 attention 和 MLM 损失权重实验。输入扰动与 MLM/NSP 训练未验证，不等同自回归聊天模型。
5. [LoRA 精读](../../notes/lora.md)：从全参数微调转向冻结权重与低秩增量，验证初始化梯度和线性合并；语言模型适配尚未运行。
6. [ViT](../../notes/vit.md)：考察图像块投影与位置标签，已有机制实验。它与 LoRA 不构成替代链。
7. [CLIP](../../notes/clip.md)：把图像与文本编码器接到配对目标，手算双向交叉熵，运行配对置换与候选集合实验；区分该目标与 BERT 的 MLM。预训练和零样本分类未复现。

## 先修与验收

先完成 LN 和矩阵乘法。对 Q、K、V 均为 3×2 的例子，写出分数矩阵、输出尺寸与 softmax 的归一化方向。

运行 `python3 experiments/run.py --experiment attention`。实验修改未来位置的 K、V，检查早期输出不变；没有词表、位置编码、训练或语言理解能力。

下一阶段（尚未实现）：带/不带因果掩码的小型序列模型。明确训练可见标签与推理可用信息，不能把泄漏未来 token 导致的低训练损失当成模型更好。

源端 padding 的独立实验：`python3 experiments/additive_attention.py`。与上面的因果掩码实验对照阅读，明确两类掩码的目的。
