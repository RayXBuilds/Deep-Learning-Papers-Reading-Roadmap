# Transformer-XL · 分段计算怎样保留上段的语言线索？

Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context。Zihang Dai、Zhilin Yang、Yiming Yang、Jaime Carbonell、Quoc V. Le、Ruslan Salakhutdinov。

2019 年 1 月预印本，同年发表于 ACL，DOI 10.18653/v1/P19-1285。来源核对日期为 2026-11-21。[原文](https://aclanthology.org/P19-1285.pdf) · [预印本记录](https://arxiv.org/abs/1901.02860) · [作者代码](https://github.com/kimiyoung/transformer-xl)

先修：[Transformer](transformer.md) 的键、查询和值，以及分段反向传播。

## 问题

将文档切成长度 L 的训练块，可把注意力矩阵控制在预算内，但每块开头可见的历史骤然缩短。提高上下文上限与补回切段处刚丢失的几个词，是可以分别检验的两种收益。

## 方法与关键推导

第 3.2 节对第 n 层查询只取当前段，用上一层的缓存与当前段拼成键和值：$H_{KV}=[\mathrm{stopgrad}(H_{old}^{n-1});H_{now}^{n-1}]$。若缓存 M 个位置，分数矩阵为 $L×(M+L)$，本段仍遮住未来。缓存保存旧隐藏表示；当前损失可以读到它，却不会沿旧段展开反向传播。更深层读取已经汇集历史的低层状态，因此前向依赖可以超过单段。

第 3.3 节将相对距离 $i-j$ 编入打分。把四项合写为 $(q_i+u)^T k_j+(q_i+v)^T r_{i-j}$，其中 k 与 r 已分别投影。两段各自从局部位置零计数时，同一编号并不说明两词相邻；相对距离给缓存中的键一个相对于当前查询仍有意义的位置。

## 原论文证据

ACL 版表 6 在 WikiText-103 的 128M 参数、Full loss、同一相对编码设置下，有递归时 PPL init 为 27.02，无递归为 29.59；这里均沿用训练时的注意长度。另看表 7，打乱句子顺序的 One Billion Word 仍有递归收益，因此切段处上下文破碎本身也值得控制。PPL 衡量预测，不直接测量模型记得哪个事实。

## 与其他论文的关系

与 [Transformer](transformer.md) 的因果预测目标一致，变化发生在可读取历史的组织和位置打分。[XLNet](xlnet.md) 复用段记忆和相对位置，同时另外改变预测目标的分解顺序。

## 可验证实验

本地未跑段缓存模型。可在相邻两段中仅改上段末词，固定查询和权重，对比启用缓存与清空缓存时下段首位的 logits，并用跨文档清空作为负对照。未测 PPL、长文检索或推理吞吐量。

## 局限与思考

缓存长度、层数和训练时截断决定能读取及能训练哪些依赖。把两篇无关文章误当成相邻段，会改变条件分布；部署时要记录跨样本文档边界怎样清理缓存。
