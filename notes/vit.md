# ViT：图像块成为 token 后，模型还知道位置吗？

[首页](../README.md) · [视觉路线](../docs/roadmaps/training-and-vision.md) · [先读 Transformer](transformer.md)

- 原题：**An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**。
- 作者：Alexey Dosovitskiy、Lucas Beyer、Alexander Kolesnikov、Dirk Weissenborn、Xiaohua Zhai、Thomas Unterthiner、Mostafa Dehghani、Matthias Minderer、Georg Heigold、Sylvain Gelly、Jakob Uszkoreit、Neil Houlsby。
- 年份与版本：首次预印本 2020-10-22；核对 [arXiv:2010.11929v2](https://arxiv.org/abs/2010.11929v2)，2021-06-03；[全文](https://arxiv.org/html/2010.11929v2)。DOI：10.48550/arXiv.2010.11929。
- 发表状态：ICLR 2021 会议论文，由 [Google Research 出版页](https://research.google/pubs/an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale/)核实。v2 更新了 CLS/GAP 讨论和计算量勘误；下文图表号按 v2。
- 官方代码：[google-research/vision_transformer](https://github.com/google-research/vision_transformer)，由论文直接链接；核对 [models_vit.py 固定版本](https://github.com/google-research/vision_transformer/blob/64801f1b3b367b3611cc27a3d45cc22870a36fb3/vit_jax/models_vit.py)。仓库包含后续模型和训练研究，不能全部倒写成原论文配置；本次未安装 JAX/Flax 或下载权重。
- 收录理由：补齐视觉路线已有条目，解释图像序列化、空间信息与数据规模的关系，不新增重复书目。
- 先修：矩阵乘法、卷积、softmax、[Transformer](transformer.md)、[LayerNorm](layer-normalization.md)、[ResNet](resnet.md)。
- 状态：中文精读初版；图像块投影与排列机制已验证；图像分类训练、预训练规模与位置插值未复现。核对日期：2026-11-02。

## 问题

图像不是天然的一维词序列。若逐像素做全局注意力，序列很长；若先用卷积提取特征，又保留了卷积的结构先验。ViT 选择把局部图像块投影成向量，再用 Transformer 编码器处理，并研究这种架构在不同预训练规模下的迁移表现（原文 §3、§4）。

理解它需要拆开三个问题：输入怎么组织、空间位置从哪里进入、分类效果由什么证据支持。实现前两项并不等于验证第三项。

## 方法与关键推导

**先追踪尺寸。** 输入 $X\in\mathbb R^{H\times W\times C}$，块边长为 $P$，假设高宽均可整除。块数 $N=HW/P^2$，每块展平为长度 $P^2C$ 的行向量；共享投影 $E\in\mathbb R^{P^2C\times D}$ 把它变成 D 维 token。加上一个可学习的分类 token 和位置表：

$$
Z_0=[x_{\rm cls};x_p^1E;\ldots;x_p^NE]+E_{\rm pos},\qquad
E_{\rm pos}\in\mathbb R^{(N+1)\times D}.
$$

这是原文式 (1) 的数据路径。以 224×224 RGB 图像、P=16 为例，196 个块各有 768 个像素分量；加入 CLS 后是 197×D，而不是把 196 个块平均为一个向量。这里的 token 仍是连续像素投影，没有词表查找或离散图像码本。

**共享投影为何可以写成卷积？** 以下是本项目的索引推导。采用块内“行、列、通道”顺序：

$$
t_{ij,d}=\sum_{u=0}^{P-1}\sum_{v=0}^{P-1}\sum_{c=0}^{C-1}
X_{iP+u,jP+v,c}\,E_{(uP+v)C+c,d}.
$$

令卷积核 $K_{u,v,c,d}=E_{(uP+v)C+c,d}$，这正是核 P×P、步长 P、无填充的深度学习卷积（严格说是互相关）。因此官方代码使用 `nn.Conv` 做 patch embedding，并不与“把块展平后线性投影”矛盾。我们的对照两侧均无偏置；若加偏置，两侧必须一致。通道优先与通道最后的展平约定也必须同步调整权重，形状一致不足以保证数值一致。

这也说明 ViT 并非完全没有图像先验：分块已经引入局部区域与固定网格。原文讨论的是相对 CNN 更少的图像专用先验，不是零先验。

**位置标签与数组顺序是两回事。** 考虑没有位置编码、掩码或 dropout 的自注意力 $F$。令 $\Pi$ 为 token 置换矩阵，各位置使用共享投影，则

$$
F(\Pi Z)=\Pi F(Z).
$$

推导：Q/K/V 同时被重排，分数矩阵变成 $\Pi QK^\top\Pi^\top$；按行 softmax 后仍是同样的行列重排，乘以 $\Pi V$ 即得到重排后的输出。这是置换**等变**；若置换固定 CLS 的位置，则 CLS 输出**不变**。逐 token 的共享 MLP、LayerNorm 和残差不会单独打破这项对称性；位置相关操作或掩码则可能改变条件。

给定位置表 $E_{\rm pos}$，比较两种干预：

| 干预 | 编码器输入 | 含义 |
|---|---|---|
| 只换内容，位置表固定 | $\Pi Z+E_{\rm pos}$ | 改变“哪个内容处在哪个位置”，CLS 输出可能变化 |
| 内容与位置一起换 | $\Pi(Z+E_{\rm pos})$ | 只是重新排列带位置标签的元素；仍满足等变关系 |

所以“加位置编码后任意重排都会改变输出”也不准确。另一个边界是：块内像素已有固定展平顺序，无位置编码的块集合不等于无序的像素集合。

**编码器与分辨率。** 原文式 (2)–(4) 使用子层前 LN，残差后接下一子层，最后对 CLS 状态作 LN；不能把共享 Transformer 名称理解为与所有原始实现逐层相同。位置表是在固定网格上学习的参数；提高分辨率但保持 P 不变会增加块数，原文 §3.2 按二维网格插值位置嵌入。本次只解释此步骤，尚未实现插值。

**计算量手算。** 224×224、P=32 时含 CLS 共 50 个 token；P=16 时为 197。单头分数矩阵元素数之比 $197^2/50^2=15.5236$。这是密集注意力分数规模的比例，不是整个模型的耗时或显存倍数；投影、MLP、实现和内存策略也影响资源。更小块带来更细的空间划分，同时增加序列成本。

## 原论文证据

以下是作者实验的阅读入口，不是本地测量：

| 位置 | 任务与对照 | 应保留的限定 |
|---|---|---|
| §4.2 表 2 | 多个分类迁移任务，ViT 与既有模型；报告微调多次运行的均值/波动 | 不同预训练数据、优化器和训练日程影响比较，不能只归因于架构 |
| §4.3 图 3、图 4 | 不同预训练数据规模；另有 JFT 子集上的少样本线性评价 | 全参数微调与少样本线性评价不是同一指标，子集实验也有早停 |
| §4.4 图 5、附录表 6 | JFT 下架构的计算量/迁移效果对照 | 使用修订版计算量，结果不构成所有规模和训练配方的普遍排序 |
| 附录 D.3 图 9、D.4 表 8 | CLS/GAP 与位置编码消融 | 分类头需结合学习率；位置表比较是特定模型的 ImageNet 5-shot linear 评价 |

这些证据支持在具体配置下讨论数据规模与先验，不支持“小数据必败”或“注意力全面替代卷积”的结论。论文的主要实验是监督分类预训练，不能因借用了 BERT 的 CLS 就把它解释为 MLM 预训练。

## 与其他论文的关系

- **[Transformer](transformer.md) → ViT：输入表示迁移。** 用连续图像块替代文本嵌入，使用编码器；图像分类通常不需要翻译解码器的因果掩码。
- **[ResNet](resnet.md) ↔ ViT：架构与数据需求对照。** 比较局部卷积组织与全局注意力时，要控制数据、增强、优化和预算；两者也能组成原文讨论的混合架构。
- **[LayerNorm](layer-normalization.md) → ViT：组件联系。** 每个 token 内的统计和子层前归一化是两个不同选择，不应混称“用了 LN 就一样”。
- **[BERT](bert.md) ↔ ViT：汇聚接口联系。** 都使用分类 token，监督任务不同；CLS 是聚合位置，不是先验写入的类别答案。
- **ViT ↔ [LoRA](lora.md)：后续适配问题。** 图像 token 的编码方式与冻结/更新哪些权重是不同维度；当前没有本仓库的 ViT-LoRA 微调证据。

## 可验证实验

```bash
python3 experiments/vit_patches.py --output /tmp/vit-patches.json
python3 -m unittest discover -s experiments -p 'test_vit_patches.py' -v
```

[协议与实测](../experiments/results/vit-patches.md) · [原始 JSON](../experiments/results/vit-patches.json)。4×4×2 人工图像，P=2、D=2；固定投影、CLS、位置表和置换，无随机数，种子不适用。投影与卷积以独立循环和手算测试交叉核对；注意力只用一个头及恒等 Q/K/V 投影，未实现完整编码器。

实际投影/卷积误差约 8.88e-16，错误展平约定产生 0.09375 差异；无位置时 CLS 重排差异为 0，固定位置只换内容时为 0.027962，内容与位置一起重排时约 6.66e-16。7 项机制检查、7 项新增语义测试通过。没有训练、数据集准确率或性能测量。

## 局限与思考

本实验说明的是等价算子与排列条件，不能证明所设位置表已经学到有用几何关系，也不能复现论文数据规模曲线。官方权重推理、位置插值、图像增强、CLS/GAP 训练对照均未运行。

可操作的后续问题：在固定数据划分和预算下，移除位置表是否改变分类准确率？比较 CLS 与 GAP 时分别调整学习率后差异是否仍存在？比较 P=16 与 P=32 时，应同时报告训练 token 数、计算量和图像分辨率，并至少运行三个种子；不能把更多计算带来的收益全部归因于空间粒度。
