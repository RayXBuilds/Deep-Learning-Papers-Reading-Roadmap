# 加性注意力机制实验 · 2026-10-28

[精读](../../notes/bahdanau-attention.md) · [实验目录](../README.md) · [机器可读结果](additive-attention.json)

## 假设与配置

动态 query 可以改变加性注意力权重；正确屏蔽 padding 后，额外存储位置不应改变上下文。进一步对照去掉 tanh 后 query 项在 softmax 中抵消的性质。

- Python：3.9.6；平台：`macOS-27.0.1-arm64-arm-64bit`；CPU、仅标准库。
- 输入：h=[[-2,4],[0,0],[2,4]]；query=[[1],[-1]]。
- 投影：Wq=[[1],[-1]]，Wh=[[1,0],[1,0]]，v=[1,-1]。
- padding：[[1000,-1000],[-2000,2000]]；有效长度分别为 3、2。
- 全部参数人工设定，无随机数，seed 不适用，没有训练/测试数据集。
- 脚本 SHA-256：`28cfe03e71145b65a40f9d6294844796fdd3932ef0b96acf91a57fb32718ffb1`。

```bash
python3 experiments/additive_attention.py --output experiments/results/additive-attention.json
python3 -m unittest discover -s experiments -p 'test_*.py' -v
```

## 实测结果与事先设定的验收条件

| 观测 | 实测 | 验收 |
|---|---:|---|
| 两个 query 的上下文最大差 | 2.095444 | > 1 |
| 固定均值上下文差 | 0.000000 | = 0 |
| 去掉 tanh 后上下文差 | 0.000000 | < 1e-12 |
| 屏蔽 padding 后上下文差 | 0.000000 | < 1e-12 |
| 有效长度 2 与截断输入的差 | 0.000000 | < 1e-12 |
| 不屏蔽 padding 的对照差 | 263.047604 | > 1 |
| 零打分器与手算均值的差 | 0.000000 | < 1e-12 |

权重归一化与 padding 零权重检查同时通过。新增 9 项测试通过，连同原 16 项共 25 项通过；包含独立手算值、退化情形、输入干预与错误形状检查。

## 解释与局限

query=1 时中间位置权重约 0.644874；query=-1 时约 0.121013。人工配置展示了动态读取能力，没有训练出语言对齐。极端 padding 是敏感性对照，不是自然数据上的性能测试；误差阈值只针对当前配置。

旧 `run.py` 和三份首轮基线未改变，此独立脚本单独记录哈希并纳入内容检查与 CI。未运行 GroundHog、WMT 翻译训练或 BLEU 复现。
