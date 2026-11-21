# XLNet · 打乱预测顺序，怎样保留词序又防止偷看答案？

XLNet: Generalized Autoregressive Pretraining for Language Understanding。Zhilin Yang、Zihang Dai、Yiming Yang、Jaime Carbonell、Ruslan Salakhutdinov、Quoc V. Le。

2019 年 6 月预印本，核对 NeurIPS 2019 会议版，arXiv v2 更新于 2020 年。来源核对日期为 2026-11-21。[原文](https://papers.neurips.cc/paper/8812-xlnet-generalized-autoregressive-pretraining-for-language-understanding.pdf) · [预印本记录](https://arxiv.org/abs/1906.08237) · [作者代码](https://github.com/zihangdai/xlnet)

先修：[Transformer-XL](transformer-xl.md) 的段记忆，以及联合概率的链式法则。

## 问题

逐词预测有规范的条件概率分解，但固定从左到右使目标词无法利用右侧词。若允许任意顺序，又必须让模型知道“接下来要预测哪一位置”，同时藏住该位置的词值。

## 方法与关键推导

第 2 节令 z 为原句下标的一个排列，最大化 $E_z[∑_t \log p(x_{z_t}∣x_{z_{<t}},z_t)]$。排列作用于概率分解次序，输入仍保留原句位置。实现中只预测排列后部的一部分词，以提供足够上下文并降低优化难度。

内容流 $h_i$ 可访问排列中截至 i 的内容，包括 i 自己，以供后续目标读取。查询流 $g_i$ 带目标位置，用严格排在 i 前面的内容作键值，自身词值不会进入预测。例：原位置为 0、1、2、3，预测顺序取 2、0、3、1。预测原位置 0 时允许读取原位置 2；预测 3 时允许读取 2、0。若把内容流直接拿来预测同一个词，对角线就会泄漏标签。两流共享注意力参数。

## 原论文证据

表 6 控制为 12 层、Wikipedia 与 BooksCorpus，报告五次运行的中位数。第 2 行 DAE + Transformer-XL 的 SQuAD 2.0 F1 为 79.56，第 3 行 XLNet-Base（K=7）为 81.33。检查缓存应另取第 4、5 行（K=6），移除缓存时 RACE 从 66.66 到 65.55。这两个比较分别约束目标选型和段记忆的解释。

## 与其他论文的关系

读 [GPT-1](gpt-1.md) 后将固定三角遮罩换成排列定义的偏序，再与 [BERT](bert.md) 对照：MLM 同时预测被选位置，XLNet 的后续目标可以读先前已纳入条件的目标内容。[Transformer-XL](transformer-xl.md) 提供段记忆与距离编码，二者都不单独产生排列目标。

## 可验证实验

四个位置的 24 种排列产生 96 个目标查询，查询流自读为 0 次，误用内容流会出现 96 次。46 个查询可看到至少一个原句右侧词。输入、验收规则、程序和输出见[实验报告](../experiments/results/pretraining-layouts.md)。

## 局限与思考

穷举验证可见性，不衡量模型学会了多少右侧信息。即使遮罩正确，错误的位置标记也可能让不同目标不可区分。可固定相同可见内容，交换两个候选的查询位置，观察训练后输出是否区分它们。
