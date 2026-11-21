# Mamba-2：如何用两种运算读取同一段历史？

Tri Dao、Albert Gu，《Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality》。2024-05-31 首稿，[ICML 2024 会议版](https://proceedings.mlr.press/v235/dao24a.html)。[版本原文](https://arxiv.org/abs/2405.21060v1) · [作者实现或发布入口](https://github.com/state-spaces/mamba)

## 问题

逐 token 推理希望只保存一个状态，训练时却希望大量位置一起做矩阵乘法。给定一串写入向量、读出向量和随位置变化的遗忘率，能否换一种求值顺序而保留每个位置的输出？

## 方法与关键推导

取单头单输入通道，令状态 $h_t\in\mathbb R^N$，初态为零。离散化后 $h_t=a_t h_{t-1}+B_t x_t$，$y_t=C_t^\top h_t$。关键限制是同一头各状态坐标共用标量 $a_t$。展开到位置 $s$，该次写入的系数为 $M_{ts}=(C_t^\top B_s)\prod_{j=s+1}^{t}a_j$；当 $s=t$ 时空积为一，未来位置系数为零。因此 $y=Mx$，即内容内积乘一个因果衰减核。它允许带符号的系数，也没有行 softmax。

在块边界切开，过去写入对后块的全部影响都要经过边界处的 $N$ 维状态，故跨边界子矩阵秩至多为 $N$。第 6 节据此将块内贡献交给矩阵乘法，只在块之间传状态。显式构造整张 $T\times T$ 矩阵适合验算，长序列实现则利用上述分解。

## 原论文证据

第 9.1 节图 8 用多键值对关联回忆比较模型，状态大小决定递推模型能保留多少历史。第 9.3 节另测内核耗时；读速度曲线须固定序列长、头宽、状态大小及硬件。召回和运行时间分别检验表示能力和实现效率。

## 与其他论文的关系

与 [Transformer](transformer.md) 对读因果混合矩阵：两者都可写成对历史值加权，SSD 对跨块信息施加有限维状态约束。任意内容相关的 softmax 注意力矩阵未必满足这一约束。[FlashAttention](flashattention.md) 也分块，但保留原注意力算子，通过逐块更新最大值与归一化分母来节省中间存储。两篇都能改进训练实现，所利用的数学结构不同。

## 可验证实验

运行 `python3 experiments/ssd_duality.py`。状态宽 3、序列长 13、种子 7/19/41。显式核对照逐步递推，最大绝对误差均为 1.11e-16（阈值 1e-12）。在第七次写入处将衰减设为零，扰动前六项后，重置前读出有变化，重置后误差均为零。[逐种子结果](../experiments/results/ssd-duality.json)

## 局限与思考

实测只覆盖标量转移算子的代数等价。若将 $a_t$ 换成每维不同的对角元素，衰减必须移入内积；可构造两个衰减速率不同的状态检查原标量核在哪一步失效。
