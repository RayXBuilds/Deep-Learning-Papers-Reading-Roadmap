# Layer Normalization：为什么不需要同批其他样本？

[首页](../README.md) · [序列路线](../docs/roadmaps/sequence-and-attention.md) · [实验说明](../experiments/README.md)

- 原题：Layer Normalization。
- 作者：Jimmy Lei Ba、Jamie Ryan Kiros、Geoffrey E. Hinton。
- 年份与版本：[arXiv:1607.06450v1](https://arxiv.org/abs/1607.06450v1)，2016-07-21；DOI：10.48550/arXiv.1607.06450。按预印本收录，同行评审发表状态未独立核实。
- 一手正文：[v1 全文](https://arxiv.org/html/1607.06450v1)；核对日期 2026-11-21。
- 官方代码：[作者仓库 ryankiros/layer-norm](https://github.com/ryankiros/layer-norm)，README 提供 LN、GRU/LSTM 接入及模型入口，依赖 Python 2.7/Theano，另有 Torch 实现。本次核对 README，未安装或运行作者训练代码，也未验证历史权重下载。
- 收录理由：连接 BN 与序列模型的关键前置；本次补全统计范围、仿射参数边界及可操作反例，不重复建文。
- 核对位置：§3 式 (3)、§3.1 式 (4)、§6.1 Table 2、§6.2 Figure 2。
- 先修：先读 [BN](batch-normalization.md)，理解二维矩阵和序列张量。
- 状态：中文精读初版；机制实验；未复现论文的 RNN 实验。

## 问题

如果某个样本的表示依赖同批其他样本，批量大小和组批方式就可能影响它。LN 将归一化统计量放在单个样本内部，使这一步计算不依赖其他训练样本。

## 方法与关键推导

对 X∈R^(B×D)，每一行独立计算 μ_b、v_b，然后 z_bd=(x_bd−μ_b)/sqrt(v_b+ε)，再应用可学习缩放和平移。对常见序列表示 B×T×D，按每个 token 的 D 维归一化时，各时间位置也分别计算；具体实现必须注明 normalized shape。

这种算子在训练和推理阶段使用相同的统计规则，不需要为它维护跨批滑动均值。但模型中的 Dropout 等其他组件仍然可以有训练/推理差异。

### 本项目推导与反例：独立的究竟是什么

对一个样本的 D 维向量 x，令 c=x−μ1、s=√(cᵀc/D+ε)，则

$$
y_i=\gamma_i c_i/s+\beta_i,\qquad
J_{ij}=\frac{\gamma_i}{s}\left(\delta_{ij}-\frac1D-\frac{c_i c_j}{D s^2}\right).
$$

这是将 [BN 的统计量求导](batch-normalization.md) 移到特征轴后的本项目推导。LN 的 γ 是逐特征向量，不能把它当作单个标量搬到所有行之外；一般 J 不对称。没有跨样本统计不等于没有特征间耦合：非对角项通常不为零。J1=0 仍成立，而 Jc 的第 i 项为 γ_i ε c_i/s³；有限 ε 下不应宣称严格尺度不变。本次只做前向反例，未数值验证逐特征 γ 的 Jacobian。

手算 x=[1,2,3]，μ=2、v=2/3。令 a=1/√(2/3+ε)，标准化输出为 [−a,0,a]。整行加 10 后输出相同；只把最后一个特征改成 4，则连第一个输出也会变化。若 γ=[1,2,3]、β=0，输出成为 [−a,0,3a]，其均值是 2a/3，已经不是零。归一化前的中间量与仿射变换后的层输出必须分开讨论。

RNN 在每个时间步独立**计算统计量**，不意味着隐藏状态独立于过去：当前输入到 LN 的向量仍含上一时刻隐藏状态。类似地，Transformer 的注意力先混合 token 信息，再按 token 的特征归一化，LN 不会撤销这种依赖。把 B×T×D 中的 T 和 D 一起归一化，则统计量会直接耦合不同位置，已经不是本文按 D 的设定。

## 原论文证据

- §3 式 (3) 定义单样本层内统计；§3.1 式 (4) 将它用于 RNN，增益与偏置跨时间共享，统计量每步重算。
- §6.1 Table 2：MSCOCO 的 order-embedding 对照中，作者自己的基线与加 LN 的 caption retrieval R@1 分别为 46.6、48.5；image retrieval R@1 为 37.8、38.9，按五个测试划分取平均。读表应使用同组基线，不混合其他论文的行。
- §6.2 Figure 2 是 attentive reader 在 CNN 问答数据上的验证结果，正文说明其数据每段限制为四句，脚注明确仅报告验证集。这不是最终测试集泛化证据，也不是所有 CNN 视觉网络的比较；此处 CNN 是新闻语料名称。

以上是原论文报告，不是本仓库测量。本文不把特定任务的收益推成 LN 全面优于 BN。

## 与其他论文的关系

BN ↔ LN 是方法对照，核心变化是统计轴。[Transformer](transformer.md) 与 LN 是组件联系，使用 LN 与残差组织子层；学习这一联系比把 LN 记成“BN 的升级版”更准确。

## 可验证实验

```bash
python3 experiments/run.py --experiment normalization
```

改变同批其他行，第一行 LN 输出应不变；检查每行归一化后的均值接近零。实验与 BN 共用输入，便于隔离统计轴的影响。

### 2026-11-21：用既有实现核对手算边界

在仓库根目录运行下面的 Python 代码；CPU、标准库，固定输入，无随机数（seed=null）、无训练或数据下载。ε=1e-5，调用现有 `run.py` 的固定仿射 LN。独立参照是上面的三元素手算；通过条件是等价项最大误差 <1e-12，单特征干预变化 >0.1，逐特征增益后的均值 >0.1。没有新增实验专题。

```python
import sys
import math
sys.path.insert(0, 'experiments')
from run import layer_norm, max_difference
x = [1., 2., 3.]
a = 1 / math.sqrt(2/3 + 1e-5)
y = layer_norm([x])[0]
shifted = layer_norm([[v + 10 for v in x]])[0]
changed = layer_norm([[1., 2., 4.]])[0]
with_peer = layer_norm([x, [90., -40., 8.]])[0]
affine_mean = sum(g*v for g, v in zip([1, 2, 3], y)) / 3
assert max_difference(y, [-a, 0, a]) < 1e-12
assert max_difference(y, shifted) < 1e-12
assert max_difference(y, with_peer) < 1e-12
assert abs(changed[0] - y[0]) > 0.1
assert abs(affine_mean - 2*a/3) < 1e-12 and affine_mean > 0.1
print('checks=5; peer_delta=', max_difference(y, with_peer))
print('feature_delta=', abs(changed[0]-y[0]), 'affine_mean=', affine_mean)
```

实际运行结果（Python 3.9.6，macOS arm64）：5 项检查通过；同批样本干预差为 0，单特征干预使第一项变化 0.15569415445809254，逐特征增益后的均值为 0.81649045727226。该代码仅扩展已有前向实验的输入与手算对照，不实现反传或训练；源脚本哈希：8d130aff2172759c00c1379dad62412bcff5c0f11854e5cb79af59f2cc8485d3。

## 局限与思考

实验只覆盖二维、固定仿射参数的前向计算；没有实现反向传播或训练性能比较。试着对同一行所有元素加常数，再只改变一个元素，说明两种改动为何不同。回答：去掉批次依赖是否等于去掉所有样本间交互？注意力层提供了怎样的反例？
