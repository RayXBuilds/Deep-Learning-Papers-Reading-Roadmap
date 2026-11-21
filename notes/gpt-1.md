# GPT-1 · 只读左侧的模型怎样完成句对分类？

Improving Language Understanding by Generative Pre-Training。Alec Radford、Karthik Narasimhan、Tim Salimans、Ilya Sutskever。

OpenAI 于 2018-06-11 公开的技术报告，按作者发布版本收录。来源核对日期为 2026-11-21。[原文](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) · [作者发布记录](https://openai.com/index/language-unsupervised/) · [作者代码](https://github.com/openai/finetune-transformer-lm)

先修：[Transformer](transformer.md) 的因果遮罩和交叉熵。

## 问题

语言模型从长文本获得监督，句对蕴含或多选题却要求固定维度的决策。问题在于怎样安排输入，让单向网络能在作决定时读完全部条件，又能继承预训练权重。

## 方法与关键推导

第 3 节先在 BooksCorpus 上最大化 $L_1=∑_t \log p(u_t∣u_{<t})$（实际截取有限上下文）。解码器式 Transformer 用因果自注意力构造每个前缀的状态。微调时在最后一个位置的状态上接线性分类头，最大化标签对数似然 $L_2$，可联合优化 $L_2+λL_1$。此时标签梯度也会更新主干。

图 1 将蕴含题编码为前提、分隔符、假设、末端读出位置。末端能看见句对双方，但前提位置的表示无法回头读取假设。多选题为每个候选构造一条序列、共享打分参数，再在候选维度归一化；任务结构通过序列排布与读出方式进入模型。

## 原论文证据

表 5 在同一组八个任务上比较多种消融。完整模型平均 74.7，去掉预训练为 59.9；保留预训练、移除微调期辅助 LM 后为 75.0。均值混合准确率、相关系数和 F1，不能解释成某一任务提高 14.8% 准确率。两行消融也说明预训练收益与辅助目标收益需要分别观察。

## 与其他论文的关系

先修是 [Transformer](transformer.md) 的因果遮罩。[ELMo](elmo.md) 学任务层混合权重，GPT-1 更新解码器本身；[BERT](bert.md) 同样可以整体微调，却通过受扰动的双向输入预训练。

## 可验证实验

本批未训练语言模型或分类器。可固定词表、句对顺序和模型，分别从末端和前提末位读出，替换假设词后检查读出向量差值。因果遮罩正确时，后一个位置选择应对假设替换保持不变。这个检查不能测量迁移精度。

## 局限与思考

句对顺序尤其重要。相似度任务在原文中显式处理两个输入顺序；蕴含关系有方向，交换前提和假设会改变问题。比较编码方式时必须同时报告构造规则。
