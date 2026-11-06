# WGAN：分布正在靠近，损失为何仍不变化？

- 原题为 Wasserstein GAN，会议版题名为 Wasserstein Generative Adversarial Networks。
- 作者为 Martin Arjovsky、Soumith Chintala、Léon Bottou。
- 首次预印本为 2017-01-26，本次核对 v3（2017-12-06）。arXiv 1701.07875，预印本 DOI 为 10.48550/arXiv.1701.07875。[版本记录](https://arxiv.org/abs/1701.07875) · [全文](https://arxiv.org/html/1701.07875v3)
- 正式发表于 ICML 2017，PMLR 70:214–223，已同行评审。[会议记录](https://proceedings.mlr.press/v70/arjovsky17a.html)
- 已核实作者代码仓库及其 README。官方训练尚未运行。[官方代码](https://github.com/martinarjovsky/WassersteinGAN)
- 收录理由是接续 GAN 的 JS 推导，解释距离、critic 函数范围和训练信号的关系。
- 先修为期望、概率分布、导数和 Lipschitz 条件，可先读 [GAN](gan.md) 与 [VAE](vae.md)。
- 状态为精读初版，一维有限分布机制实验已验证，神经网络与 LSUN 训练未复现。核对日期为 2026-11-06。

## 问题

如果生成分布整体向真实分布移动，学习目标是否能体现这种接近？原文 §2 的平行线例子把这一问题具体化，§3 再把运输距离转为可训练的函数期望差。[全文](https://arxiv.org/html/1701.07875v3)

在已有 GAN 笔记中，最优判别器对应 JS 散度。本篇继续追问分布变化与损失变化的关系，并检查学习到的 critic 与理想距离之间还隔着哪些条件。

## 方法与关键推导

### 一个可以完全手算的例子

以下是本项目对原文 Example 1 的一维简化。真实分布 p=δ₀，生成分布 q=δθ，即所有概率质量分别位于 0 与 θ。只要 θ≠0，两者支持集分离。使用与 GAN 笔记一致的标准定义

```text
m = (p+q)/2
JS(p,q) = [KL(p||m)+KL(q||m)]/2
```

每个非零质量点在混合分布中的概率都为 1/2，因此 JS=log 2，与 θ 的大小无关。只有 θ=0 时变为零。阅读原文时需要统一归一化约定，v3 §2 的 JS 定义显示式省略了 1/2，但 Example 1 和后文使用 log 2 的尺度。本项目的公式与代码显式保留 1/2。

运输则必须把单位质量从 0 移到 θ，最小代价为 |θ|。当 θ 从 1 缩小到 0.001，W₁ 同步缩小，JS 在这个例子中保持 log 2。W₁ 在 θ=0 有尖点，梯度讨论应限定在非零点。

### 从运输计划到 critic

原文 Eq. (1) 与 Eq. (2) 给出运输形式和对偶形式。Π(p,q) 是边缘分别为 p、q 的联合分布集合。

```text
W₁(p,q) = inf_{γ∈Π(p,q)} E_γ[||x-y||]
         = sup_{Lip(f)≤1} (E_p[f(x)] - E_q[f(x)])
```

对上面的点质量例子，θ>0 时选 f(x)=-x，θ<0 时选 f(x)=x。二者都是 1-Lipschitz，期望差恰好等于 |θ|，所以我们能同时给出原始运输代价与达到该值的函数。

为什么限制函数变化率？自己的负对照是 θ=0.3，若任意放大斜率到 -10，差值会从 0.3 增至 3。允许无限放大时就失去这个距离尺度。另一个负对照是 p 在 -1、1 各占一半，q 全部位于 0。所有线性函数的期望差为零，但运输代价为 1。这说明只优化过窄的函数族也可能漏掉分布差异。

原文用实值 critic f_w 最大化期望差，生成器最小化 `-E_z[f_w(Gθ(z))]`。参数裁剪把 critic 参数限制在紧集内，提供某个与架构相关的 Lipschitz 上界。该上界及函数容量会影响损失尺度，实际 critic 是否接近最优也需要检查。[§3](https://arxiv.org/html/1701.07875v3)

### 复现时应记录的训练选择

Algorithm 1 使用 RMSProp、权重裁剪和多次 critic 更新。作者 README 还说明启动阶段及周期性增加 critic 更新次数的代码安排。复现应记录实际代码版本和更新日程，再比较学习曲线。[官方说明](https://github.com/martinarjovsky/WassersteinGAN/blob/master/README.md)

## 原论文证据

| 位置 | 可核对的内容 | 本仓库阅读重点 |
|---|---|---|
| §2 Example 1 / Figure 1 | 支持集移动时不同距离的变化 | 原例是二维平行线，本地例是点质量 |
| §3 Eq. (2)、Algorithm 1 | 对偶函数约束及训练步骤 | critic 容量、约束尺度和优化程度 |
| §4.1–4.3 / Figures 3–7 | LSUN Bedrooms 的 64×64 图像实验，比较 DCGAN 与 MLP 配置 | 损失趋势与样本质量的对应关系依赖配置 |
| §3 裁剪讨论、§4.2 | 裁剪与跨 critic 比较的局限 | 理想距离和实际估计之间的误差 |

这些是作者实验与理论的位置，本地数值见下一节。[原文](https://arxiv.org/html/1701.07875v3)

## 与其他论文的关系

- [GAN](gan.md) → WGAN 属于问题延续。接续最优判别器与 JS 的分析，更换分布差异的表达及 critic 约束。GAN 的非饱和 logit 梯度实验与这里的理想分布距离分析处于不同层次。
- [VAE](vae.md) ↔ WGAN 属于目标对照。VAE 以似然下界组织训练，WGAN 用受约束函数的期望差。比较结果时需要各自匹配的评价指标。
- [DDPM](ddpm.md) ↔ WGAN 属于监督来源对照。已知加噪过程与学习中的 critic 提供不同训练信号，模型、数据和预算需要分别记录。

## 可验证实验

CPU、标准库、确定性输入，seed=null。一维等权经验分布用排序配对求运输代价，JS 复用 GAN 实验的概率表计算。语义测试通过枚举小规模全部匹配方案独立核验运输最小值。

```bash
python3 experiments/wasserstein_geometry.py --output /tmp/wasserstein-geometry.json
python3 -m unittest discover -s experiments -p 'test_wasserstein_geometry.py' -v
```

实测 θ=±1、±0.1、±0.001 时 W₁=|θ|、JS=0.693147180560。θ=0 时二者为零。θ=0.3 的有限差分导数约为 1，等均值负对照的线性 critic 差为 0、W₁=1。7 项机制检查通过。[协议与实测](../experiments/results/wasserstein-geometry.md) · [JSON](../experiments/results/wasserstein-geometry.json)

## 局限与思考

本地实验覆盖有限分布几何、可手算 critic 和局部差分，未训练神经网络、执行参数裁剪或验证图像生成。点质量的精确距离不能代表高维样本上有限 critic 的估计误差。

下一步可以固定二维数据和生成器，逐步增加 critic 容量，记录目标差、独立运输参照及梯度方向。应先明确数值尺度和优化预算，再检查“更低的 critic 差是否表示更接近真实分布”。
