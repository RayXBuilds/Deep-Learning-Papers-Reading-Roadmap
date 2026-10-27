# 序列与注意力

[路线入口](README.md) · [论文索引与原文](../papers.md)

1. Encoder–Decoder：画出输入表示与逐步解码。原始序列模型资料在历史目录。
2. Bahdanau Attention：比较固定长度表示与每步动态读取源端状态。仅原文入口，精读待补。
3. Transformer：完成 [精读](../../notes/transformer.md) 和 attention 实验，区分 self-attention、cross-attention、因果掩码。
4. BERT：比较双向与因果可见性、预训练目标。仅原文入口，精读待补；不要直接等同自回归聊天模型。
5. ViT / CLIP / LoRA：分别考察输入表示、监督信号、可训练参数。三者不构成替代链，精读与实验待补。

## 先修与验收

先完成 LN 和矩阵乘法。对 Q、K、V 均为 3×2 的例子，写出分数矩阵、输出尺寸与 softmax 的归一化方向。

运行 `python3 experiments/run.py --experiment attention`。实验修改未来位置的 K、V，检查早期输出不变；没有词表、位置编码、训练或语言理解能力。

下一阶段（尚未实现）：带/不带因果掩码的小型序列模型。明确训练可见标签与推理可用信息，不能把泄漏未来 token 导致的低训练损失当成模型更好。
