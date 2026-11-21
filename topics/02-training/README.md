# 训练、归一化与优化

[全部专题](../README.md) · [首页](../../README.md) · [年份索引](../../docs/roadmaps/timeline.md)

先读 Dropout、BN/LN 与 Adam，再比较 AdamW 和精确注意力计算。

## 中文精读与阅读顺序

- [Dropout：什么量能在测试时用一次缩放估计？](notes/dropout.md)
- [Batch Normalization：统计量从哪里来？](notes/batch-normalization.md)
- [Layer Normalization：为什么不需要同批其他样本？](notes/layer-normalization.md)
- [Adam：梯度历史怎样变成参数步长？](notes/adam.md)
- [AdamW：L2 正则为什么不等于 Adam 的权重衰减？](notes/adamw.md)
- [FlashAttention：一行 softmax 如何跨块归一化？](notes/flashattention.md)

## 原始资料

以下按历史文件名列出全部资料，文件字节保持不变。文件名可能缩写，不能据此推定版本或精读状态；同主题新笔记在上方，逐篇书目校订仍需核对原文。

- [Batch Normalization_ Accelerating Deep Network Training b](<papers/model-components/Batch%20Normalization_%20Accelerating%20Deep%20Network%20Training%20b.pdf>) · [同题精读](notes/batch-normalization.md)
- [Binarized Neural Networks_ Training Neural Networks with Weights and Activations Constrained to+ 1 or−1](<papers/model-components/Binarized%20Neural%20Networks_%20Training%20Neural%20Networks%20with%20Weights%20and%20Activations%20Constrained%20to%2B%201%20or%E2%88%921.pdf>)
- [Decoupled Neural Interfaces using Synthetic Gradients](<papers/model-components/Decoupled%20Neural%20Interfaces%20using%20Synthetic%20Gradients.pdf>)
- [Dropout_ A Simple Way to Prevent Neural Networks from](<papers/model-components/Dropout_%20A%20Simple%20Way%20to%20Prevent%20Neural%20Networks%20from.pdf>) · [同题精读](notes/dropout.md)
- [Improving neural networks by preventing](<papers/model-components/Improving%20neural%20networks%20by%20preventing.pdf>)
- [Layer Normalization](<papers/model-components/Layer%20Normalization.pdf>) · [同题精读](notes/layer-normalization.md)
- [Net2net_ Accelerating learning via knowledge transfer](<papers/model-components/Net2net_%20Accelerating%20learning%20via%20knowledge%20transfer.pdf>)
- [Network Morphism](<papers/model-components/Network%20Morphism.pdf>)
- [Adam_ A method for stochastic optimization](<papers/optimization/Adam_%20A%20method%20for%20stochastic%20optimization.pdf>) · [同题精读](notes/adam.md)
- [Deep compression_ Compressing deep neural network with pruning, trained quantization and huffman coding](<papers/optimization/Deep%20compression_%20Compressing%20deep%20neural%20network%20with%20pruning%2C%20trained%20quantization%20and%20huffman%20coding.pdf>)
- [Learning to learn by gradient descent by gradient descent](<papers/optimization/Learning%20to%20learn%20by%20gradient%20descent%20by%20gradient%20descent.pdf>)
- [On the importance of initialization and momentum in deep learning](<papers/optimization/On%20the%20importance%20of%20initialization%20and%20momentum%20in%20deep%20learning.pdf>)
- [SqueezeNet_ AlexNet-level accuracy with 50x fewer parameters and_ 1MB model size](<papers/optimization/SqueezeNet_%20AlexNet-level%20accuracy%20with%2050x%20fewer%20parameters%20and_%201MB%20model%20size.pdf>)

## 状态与下一步

压缩、合成梯度、网络变形和元优化仍待精读。

[实验指南与实测边界](../../experiments/README.md) · [迁移对照](../../docs/archive/path-migration.md)
