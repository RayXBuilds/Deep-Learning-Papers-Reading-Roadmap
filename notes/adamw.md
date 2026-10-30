# AdamW：L2 正则为什么不等于 Adam 的权重衰减？

[首页](../README.md) · [先读 Adam](adam.md) · [训练路线](../docs/roadmaps/training-and-vision.md) · [实测报告](../experiments/results/adamw-decay.md)

- 原题：**Decoupled Weight Decay Regularization**；早期题名 **Fixing Weight Decay Regularization in Adam**，按同一论文去重。
- 作者：Ilya Loshchilov、Frank Hutter。
- 年份与版本：首次预印本 2017-11-14；核对 [arXiv:1711.05101v3](https://arxiv.org/abs/1711.05101v3)，2019-01-04。DOI：10.48550/arXiv.1711.05101。
- 发表状态：ICLR 2019 会议论文，arXiv 元数据与作者代码 README 均注明。另附 [OpenReview 入口](https://openreview.net/forum?id=Bkg6RiCqY7)；本次该页返回浏览器验证，未据此声称读取审稿记录。
- 一手来源：[论文全文](https://arxiv.org/html/1711.05101v3)、[作者代码](https://github.com/loshchil/AdamW-and-SGDW)。代码链接由论文直接提供；README 使用 Torch/Lua 与 GPU 图像训练，本次未安装或运行该环境。
- 收录理由：补全维护队列中已有条目的精读，连接 Adam、正则化和训练预算；不作为近七天新论文宣传。
- 先修：梯度下降、[Adam 的矩估计与偏差校正](adam.md)、逐元素运算、L2 范数。
- 状态：中文精读初版；标准库 CPU 机制实验已验证；未复现 CIFAR-10 / ImageNet32x32 泛化指标。核对日期：2026-10-30。

## 问题

给损失加上 $c\|\theta\|^2/2$，梯度就增加 $c\theta$。直接把参数乘以一个小于 1 的数，也会把参数拉向零。两个操作在什么条件下等价？如果梯度经过历史统计和逐坐标缩放，正则项还会保持同样的作用吗？

应先比较更新方程，再比较实验指标。仅看到 API 参数名叫 `weight_decay`，无法判断它进入了梯度、动量还是独立的参数更新。

## 方法与关键推导

**无动量 SGD 的基线。** 令 $g_t=\nabla f_t(\theta)$，当前学习率为 $h_t$，L2 系数为 $c$：

$$
\theta' = \theta-h_t(g_t+c\theta)=(1-h_tc)\theta-h_tg_t.
$$

所以此处每步衰减比例为 $h_tc$。原文 §2 命题 1 用每步衰减比例 $\lambda$ 表示，对应 L2 系数 $\lambda/\alpha$。必须先统一符号；这条推导不能直接推广到把正则梯度累积进动量的实现。

**Adam 中的分岔。** Adam+L2 用 $\widetilde g_t=g_t+c\theta_{t-1}$ 更新一阶矩和二阶矩，因而正则项既参与分子，又影响分母及后续历史。AdamW 的矩估计只使用数据梯度 $g_t$，然后执行：

$$
\theta_t=(1-h_tw)\theta_{t-1}
-h_t\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
$$

这里采用[已核对的 PyTorch 2.15 文档](https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html)的 `weight_decay=w` 约定；实验是本项目的标准库实现，没有调用 PyTorch。原文 Algorithm 2 的写法为 $\theta_t=\theta_{t-1}-\eta_t(\alpha u_t+\lambda\theta_{t-1})$。取 $h_t=\eta_t\alpha$、$w=\lambda/\alpha$ 才得到同一更新。不能直接把论文中的 $\lambda$ 数值抄成框架的 `weight_decay`。

“解耦”指衰减不进入自适应矩估计，不意味着在这个 API 约定下衰减量不乘学习率。若数据梯度和初始动量全为零，经过多步后：

$$
\theta_T=\theta_0\prod_{t=1}^{T}(1-h_tw).
$$

该关系给出可以手算的调度检查；常用收缩情形还要求 $0\le h_tw<1$。若已有非零动量，即使当前数据梯度为零，也仍可能有梯度历史导致的移动。

**本项目的首步反例。** 取 $\theta=(1,10)$、$g=(0,0)$、零初始矩、$h=0.1$、$c=w=0.1$。AdamW 得 $(0.99,9.9)$，两坐标都缩小 1%。Adam+L2 的有效梯度为 $(0.1,1)$，首步偏差校正后 $\widehat m=\widetilde g$、$\widehat v=\widetilde g^2$，两坐标都近似减去 0.1，得到约 $(0.9,9.9)$。相同超参数数值不会让两种更新相同。

另外，衰减应使用旧参数；先算完整梯度更新、再把结果整体缩小，会额外缩小梯度步。配套测试专门放入非零梯度检查这个差异。

## 原论文证据

- §2 命题 1–2、Algorithms 1–2：等价条件、非等价情形与解耦位置；附录 A 给出推导。
- §4.1–4.2、图 1–2：CIFAR-10 上改变学习率调度并扫描超参数；图 3 与附录 D 补充不同训练预算、ImageNet32x32 等结果。
- 实证支持的是这些架构、数据和搜索设置下的结论，不是“AdamW 总比 SGD 或 Adam 好”。作者代码 README 还包含继承的 Shake-Shake 项目介绍，不能把其中所有结果都当成 AdamW 的独立证据。

## 与其他论文的关系

- **[Adam](adam.md) → AdamW：优化与正则化。** 保留矩估计和偏差校正，改变正则信号进入更新的位置；AdamW 不是移除了自适应步长。
- **[ResNet](resnet.md) ↔ AdamW：实验载体联系。** 原论文用残差图像网络检验优化选择，不能把架构和优化器的收益混为一项。
- **[VAE 输入跳跃](vae-input-skip.md) ↔ AdamW：参数化与更新规则的联系。** 前者讨论等价生成分布的参数缩放；本篇说明实际更新还取决于矩状态及衰减约定。这是教学联系，不是二者直接继承的认定。

## 可验证实验

```bash
python3 experiments/adamw_decay.py --output /tmp/adamw-decay.json
python3 -m unittest discover -s experiments -p 'test_adamw_decay.py' -v
```

固定人工输入，检查首步手算、相同外部梯度流下的矩状态、变学习率时的累计衰减、无动量 SGD 的等价式。再在同一二次目标上，以相同初始化、200 次更新和学习率比较 Adam、Adam+L2、AdamW；保存每一步参数与未加正则的目标值。实验无随机数，种子不适用；没有外部数据或训练/测试划分。[报告](../experiments/results/adamw-decay.md)明确区分机制验收与轨迹观察。

## 局限与思考

相同外部梯度流只用于隔离矩更新；真实训练中两种方法的参数一旦不同，后续数据梯度也可能不同。相同学习率和迭代数不代表公平的最优性能比较，本实验不比较泛化，也不搜索最佳衰减。

没有实现参数分组、稀疏梯度、AMSGrad、混合精度或 normalized weight decay。框架对“未提供梯度”的处理不在本实验范围；这里始终显式传入梯度向量。

可操作问题：先把衰减设为零，检查两条实现是否逐步一致；再将更新次数翻倍，在零数据梯度条件下推导保持总衰减所需的新系数。最后解释为何这个系数变换不能直接保证神经网络训练结果相同。
