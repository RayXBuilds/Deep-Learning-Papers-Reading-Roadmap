# 生成模型

[路线入口](README.md) · [论文索引与原文](../papers.md)

先修：条件概率、正态分布、期望、KL 散度与最大似然。VAE、GAN、DDPM 是不同建模选择，不是先后替代的版本。

| 方法 | 阅读问题 | 状态 |
|---|---|---|
| VAE | 为什么用近似后验和变分下界？重参数化把随机性放在哪里？ | [精读](../../notes/vae.md) 与高斯目标机制实验 |
| VAE 输入跳跃（评价专题） | 重构误差为什么不等于生成质量？ | [精读](../../notes/vae-input-skip.md) 与共享密度对照实验 |
| GAN | 判别器提供什么信号？如何评价训练平衡和样本覆盖？ | [精读](../../notes/gan.md) 与有限概率表、生成器梯度实验 |
| WGAN | 分布靠近时距离如何变化？critic 容量与尺度怎样影响估计？ | [精读](../../notes/wgan.md) 与一维运输机制实验 |
| EBGAN | 能量、间隔和表示方向分别提供什么约束？ | [精读](../../notes/ebgan.md) 与有限分布、PT 机制实验 |
| DDPM | 固定加噪如何产生监督？预测噪声如何用于逆向生成？ | [精读](../../notes/ddpm.md) 与一维机制实验 |

先读 [VAE 的潜变量与目标](../../notes/vae.md)，再用[输入跳跃专题](../../notes/vae-input-skip.md)检查评价指标；用 GAN 的对抗训练作对照，再读 DDPM 第 2、3 节。无需先完整复现 VAE 和 GAN 才开始 DDPM。

## 实验验收

EBGAN 的 PT 专题继续检查批大小与表示维数的关系，并手算重复表示达到下界的构造。[推导与实验](../../notes/ebgan.md)

GAN 之后也可读 EBGAN，运行 `python3 experiments/ebgan_energy.py`，检查单方最优与联合均衡的条件，以及间隔和表示方向的几何对照。[实测](../../experiments/results/ebgan-energy.md)

GAN 之后可读 WGAN，并运行 `python3 experiments/wasserstein_geometry.py`，检查排序运输、JS 和线性 critic 的负对照。[实测](../../experiments/results/wasserstein-geometry.md)

运行 `python3 experiments/gan_objective.py`，核对最优判别器与 JS 等式，并对两种生成器损失做有限差分。进一步训练仍待实现。[实测报告](../../experiments/results/gan-objective.md)

运行 `python3 experiments/vae_objectives.py`，比较解析 ELBO、三个种子的采样估计与固定噪声差分梯度，再检查重构缩放与密度不变性。[实测与边界](../../experiments/results/vae-objectives.md)已记录；未训练二维 VAE。

运行 diffusion，比较经验均值/方差和理论值，在独立测试样本上比较线性噪声预测器与零预测基线。训练、测试为同分布独立抽样。

解释边界：一维高斯不是自然图像；单时刻去噪不是完整扩散采样；用真实噪声反解数据只能检查公式。

下一阶段（尚未实现）：二维混合分布、时间条件小网络、完整逆向链；报告分布覆盖、多个种子和采样预算。
