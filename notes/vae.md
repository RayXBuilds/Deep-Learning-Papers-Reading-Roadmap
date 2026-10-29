# VAE：为什么重构项之外，还需要变分下界？

[首页](../README.md) · [生成路线](../docs/roadmaps/generative-models.md) · [配套实测](../experiments/results/vae-objectives.md)

- 原题：**Auto-Encoding Variational Bayes**。
- 作者：Diederik P. Kingma、Max Welling。
- 年份与版本：首次预印本 2013-12-20；本文核对 [arXiv:1312.6114v11](https://arxiv.org/abs/1312.6114v11)，2022-12-10；[ICLR 2014 会议接收目录](https://iclr.cc/archive/2014/old-site/conference-proceedings.html)可核实正式发表身份。DOI：10.48550/arXiv.1312.6114。
- 一手全文：[HTML](https://arxiv.org/html/1312.6114v11)；另见[作者介绍](https://dpkingma.com/blog/sgvb-autodiff-efficient-magic)。本次未核实到原论文的官方实现，不将第三方教程或后续论文代码标为官方代码。
- 收录理由：补齐既有 VAE 条目的精读，为 DDPM 的变分目标及生成模型评价提供前置知识；不是新发现的论文。
- 先修：条件概率、期望、正态密度、KL 散度、链式法则。
- 状态：中文精读初版；闭式高斯与重参数化机制已验证；二维潜空间训练、图像基准复现尚未完成。核对日期：2026-10-29。

## 问题

生成时从先验抽取潜变量，再从条件分布抽取观测；训练时只有观测，需要推断潜变量。复杂解码器通常使边缘概率积分和真实后验难以直接计算。一个共享编码器为不同样本输出近似后验，避免逐样本重新求解整套推断问题。

阅读时先区分三个对象：生成模型的先验 $p(z)$、给定潜变量的似然 $p_\theta(x\mid z)$、辅助推断分布 $q_\phi(z\mid x)$。编码器输出的是分布参数，不是把每张图永久绑定到唯一潜变量。

## 方法与关键推导

原文 §2.2 式 (1)–(3) 的核心关系为：

$$
\log p_\theta(x)=\mathcal L(x)+D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p_\theta(z\mid x)),
$$

$$
\mathcal L(x)=\mathbb E_q[\log p_\theta(x\mid z)]-D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z)).
$$

下界与证据的间隙是“近似后验到真实后验”的 KL；目标中的正则项则是“近似后验到先验”的 KL。这两个 KL 不能混用。最大化下界也不保证每一步更新都增加真实边缘似然，因为间隙会随参数变化。

对角高斯设 $x\in\mathbb R^D$、$z,\mu,\ell\in\mathbb R^K$，其中 $\ell=\log\sigma^2$。原文 §3 式 (10) 对应：

$$
z=\mu+\exp(\ell/2)\odot\epsilon,\quad \epsilon\sim\mathcal N(0,I),\qquad
D_{\mathrm{KL}}(q\Vert\mathcal N(0,I))=\tfrac12\sum_j(\mu_j^2+e^{\ell_j}-1-\ell_j).
$$

重参数化把随机数来源固定为与参数无关的 $\epsilon$，保留从损失经 $z$ 到 $\mu,\ell$ 的可微路径。若实现把 `logvar` 当标准差、使用 `exp(logvar)` 作为标准差，方差会平方一次；配套差分测试专门约束这条路径。

**本项目的推导练习：用可解模型检查“下界”的含义。** 设标量 $z\sim N(0,1)$，$x\mid z\sim N(az,s^2)$，$q=N(\mu,v)$。通过配方可以得到：

$$
p(x)=N(0,a^2+s^2),\quad p(z\mid x)=N\!\left(\frac{ax}{a^2+s^2},\frac{s^2}{a^2+s^2}\right),
$$

$$
\mathbb E_q\log p(x\mid z)=-\tfrac12\left[\log(2\pi s^2)+\frac{(x-a\mu)^2+a^2v}{s^2}\right].
$$

这给出独立于采样的检查标准：任意 $q$ 的精确 ELBO 不超过 $\log p(x)$，取真实后验时二者相等。有限次 Monte Carlo 的下界**估计值**可能高于真实边缘似然；不能要求每次噪声抽样都满足不等式。

另一个易错点是高斯观测的负对数似然含有 $\log s^2$ 项，以及被 $s^2$ 缩放的平方误差。比较不同观测方差时，仅列 MSE 会丢掉改变后的归一化项。下一篇[输入跳跃精读](vae-input-skip.md)将这个问题变成可检验反例。

## 原论文证据

- §2.2–2.4：下界分解、SGVB 估计器与重参数化条件；§3、附录 B：高斯特例和解析 KL。
- §5、图 2：MNIST / Frey Face 上比较 AEVB 与 wake-sleep 的下界；图 3 比较边缘似然估计，含 Monte Carlo EM 对照。附录 A 展示低维潜空间。
- 这些证据来自原文特定数据、网络和预算；本页不把它们扩展成现代生成模型排名。下方结果是本地标量实验，不能与图像任务直接比数值。

## 与其他论文的关系

- **VAE ↔ [DDPM](ddpm.md)：目标对照。** 两者都可用潜变量和变分分解理解；VAE 此处学习编码器分布，DDPM 使用预设前向加噪链。DDPM 的噪声 MSE 也不能直接当成生成质量。
- **VAE → [输入跳跃论文](vae-input-skip.md)：评价问题延续。** 先写出归一化似然，再比较重构和生成，避免只看编码器参与的重构误差。
- **[Adam](adam.md)：实现选择。** 优化器影响更新轨迹，不负责证明 ELBO；原 VAE §5 使用的是 Adagrad，不能把后来的常用配置倒写成原论文设置。

## 可验证实验

```bash
python3 experiments/vae_objectives.py --output /tmp/vae-objectives.json
python3 -m unittest discover -s experiments -p 'test_vae_objectives.py' -v
```

固定 $x=1.2,\mu=0.4,\ell=-0.7,a=1.5,s^2=0.7$，种子 7 / 42 / 2026 各抽取 100,000 个独立高斯噪声。分别保存重构对数似然、解析 KL 和 ELBO，与闭式期望对照；固定同一噪声时用有限差分检查路径梯度。[协议与结果](../experiments/results/vae-objectives.md)列出误差容忍和所有种子，未训练模型，不存在训练/测试划分或泛化结论。

## 局限与思考

本实验选择可解线性模型来检查公式，没有展示神经网络近似后验的表达瓶颈或优化难度。它不验证图像质量、后验坍塌、可解释潜空间。后续二维训练实验仍需独立训练/测试集和多个种子。

可操作问题：把近似后验均值移离精确后验，保持生成模型不变，边缘似然与 ELBO 哪个变化？再把噪声方差加倍，比较 MSE 与完整负对数似然；解释二者为何可能给出不同结论。
