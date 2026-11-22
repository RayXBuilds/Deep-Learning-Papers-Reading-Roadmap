# 序列与注意力

[路线入口](README.md) · [论文索引与原文](../papers.md)

1. 从 [word2vec](../../topics/04-language/notes/word2vec.md) 的词表查找开始，再读 [Seq2Seq](../../topics/04-language/notes/seq2seq.md)。画出一个词向量经过递归状态影响首个译词的路径，再比较反转源句前后的距离。
2. [Bahdanau Attention](../../topics/04-language/notes/bahdanau-attention.md)：比较固定上下文与动态读取，运行加性打分、非线性消融和变长 padding 实验。
3. Transformer：完成 [精读](../../topics/04-language/notes/transformer.md) 和 attention 实验，区分 self-attention、cross-attention、因果掩码。
4. [BERT 精读](../../topics/04-language/notes/bert.md)：区分双向可见性、输入扰动与损失位置；已有 attention 和 MLM 损失权重实验。输入扰动与 MLM/NSP 训练未验证，不等同自回归聊天模型。
5. [LoRA 精读](../../topics/10-transfer/notes/lora.md)：从全参数微调转向冻结权重与低秩增量，验证初始化梯度和线性合并；语言模型适配尚未运行。
6. [ViT](../../topics/03-vision/notes/vit.md)：考察图像块投影与位置标签，已有机制实验。它与 LoRA 不构成替代链。
7. [CLIP](../../topics/07-multimodal/notes/clip.md)：把图像与文本编码器接到配对目标，手算双向交叉熵，运行配对置换与候选集合实验；区分该目标与 BERT 的 MLM。预训练和零样本分类未复现。

## 从上下文到预训练标签

1. 比较 [ELMo](../../topics/04-language/notes/elmo.md) 的层混合与 [GPT-1](../../topics/04-language/notes/gpt-1.md) 的末端读出。在句对任务上画出标签的梯度分别到达哪些参数。
2. 从 [Transformer-XL](../../topics/04-language/notes/transformer-xl.md) 画出前段缓存到后段查询的边，再读 [XLNet](../../topics/04-language/notes/xlnet.md)。前者改变历史长度，后者还改变一个目标可以读取哪些位置。
3. 把同一对缺词分别写成 [BERT](../../topics/04-language/notes/bert.md) 的分类标签与 [T5](../../topics/04-language/notes/t5.md) 的解码目标。指出哪一步允许后一缺词依赖前一缺词。

## 先修与验收

先完成 [LN 精读](../../topics/02-training/notes/layer-normalization.md) 和矩阵乘法，区分统计量每步独立计算与隐藏状态的时间依赖；运行笔记内 5 项前向手算检查。对 Q、K、V 均为 3×2 的例子，写出分数矩阵、输出尺寸与 softmax 的归一化方向。

运行 `python3 experiments/run.py --experiment attention`。实验修改未来位置的 K、V，检查早期输出不变；没有词表、位置编码、训练或语言理解能力。

下一阶段（尚未实现）：带/不带因果掩码的小型序列模型。明确训练可见标签与推理可用信息，不能把泄漏未来 token 导致的低训练损失当成模型更好。

源端 padding 的独立实验：`python3 experiments/additive_attention.py`。与上面的因果掩码实验对照阅读，明确两类掩码的目的。

## 注意力可以直接回答“选哪一行”吗？

读 [Pointer Networks](../../topics/08-memory/notes/pointer-networks.md) 第 2.3 节对应的推导。先算一组分数的 softmax，再分别求加权上下文和指定行号的交叉熵，比较哪一步需要知道目标行号。

## 参数固定时怎样指定任务

读 [GPT-3](../../topics/04-language/notes/gpt-3.md) 表 3.2–3.3，固定模型对比示例数，再固定提示协议比较模型规模。与 [LoRA](../../topics/10-transfer/notes/lora.md) 对照，标出任务信息分别进入上下文还是参数更新。

## 从预训练到偏好学习

读完 Transformer，可用 [Chinchilla](../../topics/04-language/notes/chinchilla.md) 为固定 FLOPs 分配参数和 token，再用 [FlashAttention](../../topics/02-training/notes/flashattention.md) 追踪单层中间结果的读写。训练目标转向用户指令时，沿 [InstructGPT](../../topics/04-language/notes/instructgpt.md) → [DPO](../../topics/04-language/notes/dpo.md) 比较奖励从何而来、策略接收什么梯度。沿 LoRA → [QLoRA](../../topics/10-transfer/notes/qlora.md) 则追踪冻结权重的表示及反向传播。

## 状态、专家与推理策略

先展开 [Mamba-2](../../topics/04-language/notes/mamba-2.md) 的两步递推，找出前一项的衰减系数。再沿 [DeepSeek-V3](../../topics/04-language/notes/deepseek-v3.md) → [DeepSeek-R1](../../topics/04-language/notes/deepseek-r1.md) 追踪基础模型如何接上同题多回答的奖励。最后读 [Qwen3](../../topics/04-language/notes/qwen3.md)，辨认在哪个阶段由教师生成轨迹、在哪个阶段由学生采样。
