# LoRA：少训练哪些参数，梯度又怎样流动？

[首页](../README.md) · [序列路线](../docs/roadmaps/sequence-and-attention.md) · [先读 Transformer](transformer.md)

- 原题：**LoRA: Low-Rank Adaptation of Large Language Models**。
- 作者：Edward J. Hu、Yelong Shen、Phillip Wallis、Zeyuan Allen-Zhu、Yuanzhi Li、Shean Wang、Lu Wang、Weizhu Chen。
- 年份与版本：首次预印本 2021-06-17；本文核对 [arXiv:2106.09685v2](https://arxiv.org/abs/2106.09685v2)，2021-10-16；[全文](https://arxiv.org/html/2106.09685v2)。arXiv DOI：10.48550/arXiv.2106.09685。
- 发表状态：[ICLR 2022 正式会议 Poster](https://iclr.cc/virtual/2022/poster/6319)。本文的章节与表号按 arXiv v2，不能把预印本版本日期当成会议日期。
- 官方代码：[microsoft/LoRA](https://github.com/microsoft/LoRA)，论文直接链接；本次核对 [loralib/layers.py 的固定版本](https://github.com/microsoft/LoRA/blob/c4593f060e6a368d7bb5af5273b8e42810cdef90/loralib/layers.py)中的 `Linear`。未安装官方 PyTorch 环境或下载模型。
- 收录理由：将已有书目入口补为精读，衔接 BERT 的全参数微调与参数高效适配；不是把经典论文当成近期新发现。
- 先修：矩阵乘法与秩、链式法则、[Transformer](transformer.md)、[BERT 的预训练/微调区分](bert.md)、[Adam 的矩状态](adam.md)。
- 状态：精读初版；CPU 机制实验已验证；语言模型微调、显存和吞吐量未复现。核对日期：2026-11-01。

## 问题

预训练模型适配多个任务时，如何减少每个任务需要训练和保存的参数？先区分三个量：基础权重、任务增量、训练时中间激活。减少任务参数，不会自动按相同比例减少后两者之外的全部存储或计算。

LoRA 在冻结的线性权重旁学习一个低秩增量（原文 §4.1）。问题因此变为：有限的增量表达能力够不够完成指定任务，以及优化器怎样学习这两个因子。

## 方法与关键推导

下面统一采用列向量，并给出本项目的推导。令输入 $x\in\mathbb R^k$，基础权重 $W_0\in\mathbb R^{d\times k}$，可训练因子 $A\in\mathbb R^{r\times k}$、$B\in\mathbb R^{d\times r}$，缩放 $s=\alpha/r$：

$$
y=W_0x+sB(Ax),\qquad \Delta W=sBA.
$$

数据路径是 $k\to r\to d$，最后与 $W_0x$ 相加。只计这一个无偏置线性层，可训练参数由 $dk$ 变成 $r(k+d)$；只有 $r(k+d)<dk$ 才节省参数。例：$d=k=4096,r=8$ 时为 65,536 对 16,777,216，即 1/256。此数不能直接写成整个模型的显存比例，任务头、偏置或其他可训练模块还要另算。

**低秩限制落在增量上。** $\operatorname{rank}(BA)\le r$，但 $W_0+sBA$ 不必低秩。本项目手算反例：

$$
W_0=I_2,\quad B=\begin{bmatrix}1\\2\end{bmatrix},\quad
A=\begin{bmatrix}3&4\end{bmatrix},\quad s=1.
$$

$BA$ 的行列式为 0，而 $W_0+BA=\begin{bmatrix}4&4\\6&9\end{bmatrix}$ 的行列式为 12。因此这是受限的权重适配，不是直接把基础模型压缩为两个小矩阵。

**初始化决定第一步谁能学习。** 令 $g=\partial L/\partial y$，对上式应用链式法则：

$$
\frac{\partial L}{\partial B}=s\,g(Ax)^\top,\qquad
\frac{\partial L}{\partial A}=s\,B^\top g x^\top,\qquad
\frac{\partial L}{\partial x}=W_0^\top g+sA^\top B^\top g.
$$

当 $B=0$、$A$ 非零时，初始输出保持基础层输出，第一步的数据梯度对 $A$ 为零，对 $B$ 则可能非零。更新 $B$ 后，$A$ 才可能收到非零数据梯度。“可能”不能省略：若误差或 $Ax$ 为零，对 $B$ 的梯度也会为零。若两个因子都初始化为零，普通梯度法在此参数化下不能靠数据梯度启动。

这里说的是数据梯度。本实验没有权重衰减或历史动量；加入正则项或优化器状态后，参数是否移动还要另算，不能只看当步数据梯度。

此外，$W_0$ 冻结只是禁止更新它，并未删除 $W_0^\top g$。如果输入来自上游可训练层，误把整个基础分支放进停止梯度的区域，会改变上游训练。

**合并为何可行，又在哪些条件下失效？** 在确定性线性分支中，

$$
W_{\rm merged}=W_0+sBA,\qquad W_{\rm merged}x=W_0x+sB(Ax).
$$

合并后必须关闭独立增量分支，否则计算的是两份增量。如果训练态只对增量输入施加 dropout，则计算为 $W_0x+sBA D x$；一般不等于 $(W_0+sBA)x$。推理关闭 dropout 后再讨论合并等价。本实验还没有量化、不同精度、任务切换或内核计时，不能据此证明所有部署方式都零额外延迟。

官方 `Linear` 使用 $\alpha/r$，并在合并状态下跳过分支。原文 §4.1 写 $A$ 为高斯初始化；所核对代码实际对 `lora_A` 调用 Kaiming uniform、对 `lora_B` 置零，应读可执行语句而非仅依赖附近注释。本项目使用标准差 0.1 的高斯教学初始化，不声称等同官方训练配置。

## 原论文证据

| 一手证据位置（arXiv v2） | 观察什么 | 解释边界 |
|---|---|---|
| §4.1，式 (3)；§4.2 | 因子化、初始化、合并与选用的注意力投影 | 原论文配置不能代替所有 LoRA 实现 |
| §5.2–5.5，表 2–4 | RoBERTa / DeBERTa 的 GLUE、GPT-2 的 E2E、GPT-3 的任务对照 | 数据、模型、训练预算和可训练参数要一起读；不是当前排名 |
| §7.1，表 5 | 固定约 18M 可训练参数，比较投影选择并相应改变秩 | 同时控制总参数，不能只比较单层的 r |
| §7.2，表 6 | WikiSQL、MultiNLI 的秩消融 | 小秩有效是这些配置的经验结果，不是所有任务定理 |

上述为作者报告；本地只验证下面的代数和梯度机制。会议身份由 ICLR 页面核对；OpenReview 页面本次触发浏览器验证，没有据此推测评审内容。

## 与其他论文的关系

- **[BERT](bert.md) ↔ LoRA：适配策略对照。** 从“预训练以后更新全部参数”转到“冻结基础权重、只更新指定增量”；LoRA 本身不规定 MLM 或下一词预测目标。
- **[Transformer](transformer.md) → LoRA：组件联系。** $W_q,W_v$ 等是可适配位置，修改它们可改变注意力计算；LoRA 不代替注意力，也不消除因果或 padding 掩码。
- **[Adam](adam.md) / [AdamW](adamw.md) ↔ LoRA：优化联系。** 因子化改变优化变量；对 $A,B$ 做衰减不等于直接对 $W_0+sBA$ 做同样衰减。本次用无动量 SGD 隔离初始化机制，没有验证最优优化器。
- **[ResNet](resnet.md) ↔ LoRA：教学对照。** 都可画出相加路径，但 LoRA 此处是冻结线性映射加任务增量；不能把它直接认作普通残差网络训练问题。

## 可验证实验

```bash
python3 experiments/lora_mechanics.py --output /tmp/lora-mechanics.json
python3 -m unittest discover -s experiments -p 'test_lora_mechanics.py' -v
```

[完整协议与结果](../experiments/results/lora-mechanics.md) · [机器可读记录](../experiments/results/lora-mechanics.json)。单个人工输入，基础矩阵 2×3，秩 2；以合并线性层、手算和中心差分作参照。种子 7 / 42 / 2026 仅影响高斯初始化；每个种子做两步 SGD。没有数据集、独立测试集或泛化结论。

实际结果：合并最大误差 0，$A/B/x$ 梯度的中心差分最大误差约 $2.96\times10^{-9}$；三种子均满足首步 $A$ 梯度为零、$B$ 梯度非零，下一步 $A$ 梯度非零。双零初始化仍有正损失却无因子梯度；固定 dropout 反例与朴素合并相差 1.08。10 项机制检查、9 项新增语义测试通过。

## 局限与思考

当前没有语言模型微调、质量/预算对照、显存计量或吞吐量测量。主实验的 2×3、r=2 是便于检查导数的尺寸，其因子参数甚至多于基础矩阵；参数节省仅在另列的大尺寸算例中成立，不能把这个小实验当作效率验证。

下一步可检验：固定一个小模型、相同数据划分和 token 预算，比较全参数微调、冻结模型和多种秩的 LoRA；至少三个种子，分别报告任务指标、可训练参数、峰值内存和时间。秩改变时，明确是固定 $\alpha$ 还是固定 $\alpha/r$，以及适配层是否相同，避免把缩放或模块数量变化误归因于秩。
