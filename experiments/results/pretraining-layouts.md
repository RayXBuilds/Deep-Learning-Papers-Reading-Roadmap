# 排列预测的可见性与片段去噪的序列长度

2026-11-21，在 arm64 CPU 上用 Python 3.11.15 标准库穷举。两项实验均无训练、外部数据、数据划分或随机抽样（种子为空）。[脚本](../pretraining_layouts.py) · [独立测试](../test_pretraining_layouts.py) · [实测 JSON 与脚本 SHA-256](pretraining-layouts.json)

```bash
python3 experiments/pretraining_layouts.py > /tmp/pretraining-layouts.json
diff -u experiments/results/pretraining-layouts.json /tmp/pretraining-layouts.json
python3 -m unittest discover -s experiments -p 'test_pretraining_layouts.py'
```

## XLNet · 查询中可见的内容

依照原文第 2 节，输入为四个固定位置的所有分解次序。对每个位置读两次，查询流只读取严格前驱，内容流包含自身。通过条件为所有查询均不读取目标，所有内容流负对照都能自读，并且至少一个查询可以读取自然顺序在右侧的词。实测 24 × 4 次查询中，自读 0 次，内容流自读 96 次，可读右侧 46 次。这个计数还有独立参照：目标位置 i 在其自身及右侧共 4−i 个位置中排第一的概率为 1/(4−i)，故四个目标可读右侧的次数依次为 18、16、12、0。用固定错序和逐层前驱闭包作独立检查，检验在多层传递中自读也不能绕道发生。[XLNet 精读](../../notes/xlnet.md)

## T5 · 哨兵能否无损标识缺口

输入为六个互异的词及全部 62 种同时含可见词和缺词的布尔掩码。以完整六词重建为长度基线。在省略 EOS、每段一个哨兵并附加末哨兵的约定下，要求所有输入均能精确还原，源与目标的长度必须分别等于 N−m+r、m+r+1（m 为缺词数，r 为连续缺段数）。实测两项均为 62/62，只有 25 个标签序列严格短于全文。等量缺词相邻或分离的手算对照、含重复词的两端遮盖测试另行核验边界。[T5 精读](../../notes/t5.md)

两项结果只涉及离散结构。穷举的掩码分布与 T5 的采样器不同，也未计算注意力值、优化参数或评测任务分数。
