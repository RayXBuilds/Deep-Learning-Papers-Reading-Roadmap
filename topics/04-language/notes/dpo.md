# DPO：如何从两个回答的策略概率恢复偏好？

[所属专题与原始资料](../README.md)

- 原文：[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)。
- 作者与版本：Rafael Rafailov、Archit Sharma、Eric Mitchell、Stefano Ermon、Christopher D. Manning、Chelsea Finn。2023 年首发，NeurIPS 2023；2024 年 arXiv v3。[作者实现](https://github.com/eric-mitchell/direct-preference-optimization)。
- [会议论文记录](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html)。
- 书目核对（2026-11-21）：[会议 DOI 10.52202/075280-2338](https://doi.org/10.52202/075280-2338)。
- 先修：[InstructGPT](instructgpt.md)、约束优化与条件概率。
- 状态：机制实验已验证。

## 问题

输入固定提示 x、胜者 w、负者 l 以及冻结的参考策略。离线比较数据没有给每个回答的绝对奖励。能否保留 KL 正则的含义，同时直接用这对回答更新策略？

## 方法与关键推导

对每个 x 最大化 Eπ[r]−β KL(π‖πref)，归一化约束给出 π*(y|x)=πref(y|x)exp(r/β)/Z(x)。移项得到 r=β log(π*/πref)+β log Z(x)。在同提示的 Bradley–Terry 奖励差中，后一项相消。设 Δ=log[π(w)/πref(w)]−log[π(l)/πref(l)]，式 7 的单对损失为 −log σ(βΔ)。序列 log 概率按回答 token 求和，提示 token 不计；对 Δ 的导数 −βσ(−βΔ) 给已经分对的样本较小权重。

## 原论文证据

§4 式 4–7 给出上述替换的前提。§6 图 2 左侧同时看 IMDb 情感控制的奖励和与参考策略的 KL；只比较最高奖励会忽略偏离程度。右侧在 TL;DR 摘要评测中改变采样温度，DPO、PPO、只用胜者微调共用 SFT 起点；胜率依赖评判方式及解码设置。

例如当前策略和参考策略对两回答的赔率分别为 1∶1 和 1∶2，即便当前策略给两个回答相同概率，胜者相对参考也已提高一倍。取 β=2，预测偏好概率为 4/5。

## 与其他论文的关系

[InstructGPT](instructgpt.md) 先显式学奖励，再对新生成的回答进行策略更新。DPO 直接使用离线对，参考策略仍参与 log 比值；只增加胜者的似然会丢掉负者和参考分布两组信息。

## 可验证实验

见[离散偏好实验](../../../experiments/preference_odds.py)与[实测记录](../../../experiments/results/preference-odds.json)。从给定奖励精确构造最优分类分布，逐对与奖励差的概率对照，并用偏斜参考策略检查遗漏参考项的后果。

## 局限与思考

推导需要参考策略在候选上有正概率，且比较共享提示。若训练集中只出现少数风格的回答，得到低偏好损失并不能推断未出现答案的质量；应以新采样回答另作盲评。
