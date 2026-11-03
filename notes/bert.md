# BERT：双向上下文、遮盖输入与监督位置怎样配合？

[首页](../README.md) · [序列路线](../docs/roadmaps/sequence-and-attention.md) · [先读 Transformer](transformer.md)

- 原题：**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**。
- 作者：Jacob Devlin、Ming-Wei Chang、Kenton Lee、Kristina Toutanova。
- 年份与版本：首次预印本 2018-10-11；核对 [arXiv:1810.04805v2](https://arxiv.org/abs/1810.04805v2)，2019-05-24；[全文](https://arxiv.org/html/1810.04805v2)。
- 发表状态：[NAACL-HLT 2019 正式会议论文](https://aclanthology.org/N19-1423/)，页 4171–4186；DOI：10.18653/v1/N19-1423。不能把预印本年份写成会议年份。
- 官方代码：[google-research/bert](https://github.com/google-research/bert)，由论文直接链接。本次读取 README、数据生成与预训练损失代码；未安装原 TensorFlow 环境或下载权重。当前仓库还含后续 whole-word masking 等功能，不能全部倒写成原论文配置。
- 收录理由：补齐 Transformer 到预训练表示之间的已有条目，解释三种经常混淆的“mask”；不是新论文发现。
- 先修：注意力矩阵、softmax、交叉熵、[LayerNorm](layer-normalization.md)、训练/验证/测试划分。
- 状态：中文精读初版；复用既有可见性机制实验并实际复核；新增 MLM 损失权重机制验证；输入扰动、MLM/NSP 训练与下游基准未复现。核对日期：2026-11-03。

## 问题

句子分类和抽取式问答通常允许读取完整输入。例如判断句中一个词的含义时，右侧词语可能决定其解释。BERT 用双向 Transformer 编码器学习可迁移表示，再针对任务微调；它没有因果语言模型逐词预测的训练约束。

但如果在完整、未损坏的输入上预测每个位置原词，模型可能直接利用该位置的输入，任务变成复制。因此必须同时讨论输入怎样改变、上下文怎样流动，以及损失在哪里计算。

## 方法与关键推导

原文 §3–3.1 的流程可以拆成三层：

| 层面 | 改变什么 | 不等同于什么 |
|---|---|---|
| 输入扰动 | 选出预测位置，将部分 token 换成 `[MASK]` 或随机词，也有部分保持原样 | 不等于把这些位置从注意力中删除 |
| 注意力可见性 | 有效 token 可读取左右上下文；padding 仍需排除 | 不等于下三角因果掩码 |
| 损失位置 | 在选中的预测位置对原词计算词表交叉熵 | 不等于所有非 padding 位置都有 MLM 标签 |

先选约 15% 的 WordPiece 位置；在被选位置中，80% 换为 `[MASK]`、10% 换随机词、10% 不变（§3.1）。这些是条件概率，不是对整句各自取 80%、10%、10%；短序列还受取整、候选集合和预测数量上限影响。

**本项目的记号与推导。** 原序列为 $x$，预测位置集合为 $S$，损坏后的输入为 $\widetilde x$。忽略批次权重时，MLM 的平均损失写为：

$$
L_{\rm MLM}=-\frac{1}{|S|}\sum_{i\in S}\log p_\theta(x_i\mid\widetilde x).
$$

目标来自原序列 $x_i$，而不是替换后的 token。若某个被选位置碰巧保持原样，它仍属于 $S$，仍参与损失。实现时用 `input_id == MASK_ID` 构造监督集合会漏掉随机替换与保持原样的位置。

设隐藏输出为 $H\in\mathbb R^{n\times d}$，选取 $H_S\in\mathbb R^{|S|\times d}$，经过预测头得到 $|S|\times V$ 个 logits；softmax 沿词表轴归一化。官方 [run_pretraining.py](https://github.com/google-research/bert/blob/master/run_pretraining.py) 的 `get_masked_lm_output` 先取预测位置，使用非线性变换与 LayerNorm，再与共享词嵌入权重相乘。预测槽的 padding 通过 `label_weights` 排除；实现分母还加了小常数，公式是其教学简化。

自注意力另有自己的矩阵：

$$
A=\operatorname{softmax}(QK^\top/\sqrt{d_k}+M),\qquad H'=AV.
$$

$M$ 决定哪些 key 可被读取，和 $S$ 是不同对象。`[MASK]` 是有嵌入的输入 token，不应因此被当作 padding。仅在注意力对角线上屏蔽自身，也不能一般性阻止多层网络通过其他位置绕回原词信息；数据扰动与结构可见性不能互相替代。

**手工追踪练习（非训练结果）。** 假设输入中的位置 2 和 5 被选为预测目标，位置 2 换成 `[MASK]`，位置 5 保持原词。则损失取两个位置，标签都是替换前的词。位置 3 即使参与上下文计算，也没有本次直接 MLM 目标；它仍可能通过计算图影响损失。区分“没有直接标签”和“没有梯度路径”。

NSP 是另一项二分类任务：给定文本 A/B，判断 B 是真实后续片段还是随机抽取片段；不是生成下一句。原文输入把 token、segment 和 position embedding 相加，并使用 `[CLS]` / `[SEP]` 组织序列。微调阶段根据任务接分类或位置预测头，原论文采用全参数微调（§3.2）。

## 原论文证据

- §3.1、附录 C.2：MLM、NSP 与输入替换策略；[create_pretraining_data.py](https://github.com/google-research/bert/blob/master/create_pretraining_data.py) 的 `create_masked_lm_predictions` 可追踪原标签的保存。
- §5.1 表 5：BERT_BASE、No NSP、LTR & No NSP 等预训练消融；§4 的 GLUE、SQuAD 与 SWAG 表格给出任务结果。
- 阅读表 5 时注意，No NSP 与 LTR & No NSP 同时改变了上下文方向和预训练目标。它不能单独证明“只去掉因果掩码”能带来全部收益。下方算子实验只验证信息可见性，不复现该消融。
- ACL 页面用于核对发表身份；原文结果不代表当前排行榜。训练数据、模型规模、微调选择和是否集成都应随指标一起比较。

## 与其他论文的关系

- **[Transformer](transformer.md) → BERT：预训练任务与使用方式。** 复用编码器组件，以 MLM/NSP 预训练；并非把原翻译模型的解码器直接改名。
- **[Bahdanau Attention](bahdanau-attention.md) ↔ BERT：掩码用途对照。** 前者专题的 padding 是排除无效源位置；MLM 的输入替换是制造监督问题，两者可以同时存在。
- **[LayerNorm](layer-normalization.md) → BERT：组件联系。** 相同归一化组件不意味着预训练目标或迁移效果相同。
- **BERT ↔ [LoRA](lora.md)：适配策略对照。** 本文全参数微调与 LoRA 的低秩增量适配可对照；LoRA 专题已有代数与梯度实验，尚未运行 BERT 的真实任务适配。

## 可验证实验

```bash
python3 experiments/run.py --experiment attention --seed 7 --output /tmp/bert-visibility.json
```

复用已有 [attention 协议](../experiments/README.md)与[已存核心基线](../experiments/results/baseline-seed-7.json)，不增加独立实验计数。输入 Q/K/V 均为 3×2；修改最后位置的 K/V，比较前两个输出。因果版本最大变化应小于 1e-12，无因果掩码对照变化应大于 1。固定人工输入，不使用随机数；CLI seed=7 在本子实验中不影响结果。

2026-10-31 实际复核：因果前缀变化 **0**，无因果掩码前缀变化 **104.40667255607859**，四项验收通过。代码哈希和结果与已存基线一致，故不另存重复基线。[当日记录](../docs/daily/2026/10/2026-10-31.md)登记本次运行。

这只说明注意力可否读取右侧信息，没有词表、输入扰动、MLM 预测头、NSP 或训练，也不是 BERT 权重推理。2026-11-03 新增了下述损失层验证；输入候选选择与 80/10/10 扰动采样仍未实现。

### 新增：固定 logits 的 MLM 损失验证

```bash
python3 experiments/bert_mlm.py --output /tmp/bert-mlm.json
```

[实验协议与结果](../experiments/results/bert-mlm.md) · [机器可读记录](../experiments/results/bert-mlm.json)。将预测槽索引记为 j，词表 logits 为 z，标签为 y，槽权重为 w，官方损失的教学实现为：

$$
L=\frac{\sum_j w_j\left[\log\sum_v e^{z_{jv}}-z_{j,y_j}\right]}{\sum_j w_j+10^{-5}}.
$$

分母是有效预测权重之和，不是序列长度或补齐后的槽数。稳定实现先减去每行最大 logit，再计算 log-sum-exp；权重为零的合法槽不影响分子或分母。所有权重均为零时损失返回零，不代表这个批次提供了有效训练信号。

四个固定槽模拟遮盖、选中未替换、随机替换和 padding，前三槽权重为 1，末槽为 0。正确损失 0.5364775159；仅按遮盖槽计损失得到 0.5108205156。改变选中未替换槽的标签使损失增加 0.2310482900；改变未选槽标签、padding logits 或删除 padding 槽均不改变损失。7 项检查、6 项语义测试通过。

这是对监督位置的独立干预，不是对自然文本的统计结论；没有运行编码器或预测头。槽类型由手工指定，未验证输入扰动比例。比较标签时保持 logits 不变，因而也不应据此断言上下文 token 没有反向梯度。

## 局限与思考

原始 BERT 不提供通常的左到右序列概率分解；不能把 MLM 损失直接当成因果语言模型的 perplexity。双向阅读也不等于在任何任务上优于因果模型，两者可用信息与目标不同。

固定 logits 的标签与 padding 干预已验证；后续仍需将候选选择、输入扰动和预测头连入独立划分的小语料管线。训练比较还需多个种子、相同 token 预算和独立验证集。在这些工作完成前，不声称本项目重现了 BERT 的语言理解或预训练收益。
