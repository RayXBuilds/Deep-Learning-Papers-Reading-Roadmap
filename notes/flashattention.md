# FlashAttention：一行 softmax 如何跨块归一化？

- 原文：[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)。
- 作者与版本：Tri Dao、Daniel Y. Fu、Stefano Ermon、Atri Rudra、Christopher Ré。2022 年 arXiv v2，NeurIPS 2022。[作者实现](https://github.com/Dao-AILab/flash-attention)。
- [会议论文记录](https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract.html)。
- 书目核对（2026-11-21）：[会议 DOI 10.52202/068431-1189](https://doi.org/10.52202/068431-1189)。
- 先修：[Transformer](transformer.md)、稳定 softmax。
- 状态：机制实验已验证。

## 问题

长度 n 的单头注意力要让每个 query 与全部 key 交互。若将两两分数和 softmax 概率都写到显存，后续乘 V 又要读回这两块 n×n 数据。如何在较小的片上存储里给出每行的全局加权和？

## 方法与关键推导

对一个 query 的分数 sⱼ=q·kⱼ/√d，维护已读列的最大值 m、分母 l 和向量分子 u。读入新块 B 时先设 m′=max(m,max s_B)，再更新 l′=exp(m−m′)l+Σ_B exp(sⱼ−m′)，u′=exp(m−m′)u+Σ_B exp(sⱼ−m′)vⱼ。旧项与新项由此共享指数原点。最后返回 u/l。各块先独立 softmax 再平均，会把总概率质量不同的块赋予同样权重。反向时用保存的行统计重算概率块。

## 原论文证据

算法 1 和附录 B 的归纳论证说明结果等于稠密注意力。定理 2 在片上容量 M 的指定区间给出 Θ(n²d²/M) 次 HBM 访问。§4 的 GPU 基准区分算子计时与模型端到端训练速度；前者的收益会被模型中其余层稀释。

§4 表 1 比较从同一初始化训练 BERT-large 至 72% MLM 准确率的耗时。8 张 A100 上各跑 10 次，MLPerf 实现平均 20.0 分钟，论文实现为 17.4 分钟。这里的时间同时包含其他模型组件及其实现开销。

## 与其他论文的关系

[Transformer](transformer.md) 的 Q、K、V 和缩放保持原义。稠密版本仍考虑全部允许的 token 对；论文另列了跳过部分块的稀疏扩展，后者会改变可见位置。

## 可验证实验

见[分块归一化实验](../experiments/attention_tiles.py)和[实测记录](../experiments/results/attention-tiles.json)。CPU 双精度比较不同块宽、非整除尾块和大偏移量下的逐行输出，并保留等权平均各块的反例。

## 局限与思考

运算顺序改变会影响浮点舍入。该实验只验证前向行归约；GPU 的访存、反向重算和混合精度误差均须单独测量。
