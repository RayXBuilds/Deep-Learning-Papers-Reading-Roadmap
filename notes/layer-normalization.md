# Layer Normalization：为什么不需要同批其他样本？

[首页](../README.md) · [序列路线](../docs/roadmaps/sequence-and-attention.md) · [实验说明](../experiments/README.md)

- 原文：[Layer Normalization](https://arxiv.org/abs/1607.06450)，Ba、Kiros、Hinton，2016。
- 核对位置：第 3 节、第 3.1 节、第 6 节。
- 先修：先读 [BN](batch-normalization.md)，理解二维矩阵和序列张量。
- 状态：中文精读初版；机制实验；未复现论文的 RNN 实验。

## 问题

如果某个样本的表示依赖同批其他样本，批量大小和组批方式就可能影响它。LN 将归一化统计量放在单个样本内部，使这一步计算不依赖其他训练样本。

## 方法与关键推导

对 X∈R^(B×D)，每一行独立计算 μ_b、v_b，然后 z_bd=(x_bd−μ_b)/sqrt(v_b+ε)，再应用可学习缩放和平移。对常见序列表示 B×T×D，按每个 token 的 D 维归一化时，各时间位置也分别计算；具体实现必须注明 normalized shape。

这种算子在训练和推理阶段使用相同的统计规则，不需要为它维护跨批滑动均值。但模型中的 Dropout 等其他组件仍然可以有训练/推理差异。

## 原论文证据

第 6 节比较多个任务，尤其包括递归模型。应按任务、网络和对比方法理解结论，不能据此宣称所有 CNN 中 LN 都优于 BN。

## 与其他论文的关系

从 BN 到 LN 的核心变化是统计轴。[Transformer](transformer.md) 使用 LN 与残差作为子层组件；学习这一联系比把 LN 记成“BN 的升级版”更准确。

## 可验证实验

```bash
python3 experiments/run.py --experiment normalization
```

改变同批其他行，第一行 LN 输出应不变；检查每行归一化后的均值接近零。实验与 BN 共用输入，便于隔离统计轴的影响。

## 局限与思考

实验只覆盖二维、固定仿射参数的前向计算；没有实现反向传播或训练性能比较。试着对同一行所有元素加常数，再只改变一个元素，说明两种改动为何不同。回答：去掉批次依赖是否等于去掉所有样本间交互？注意力层提供了怎样的反例？
