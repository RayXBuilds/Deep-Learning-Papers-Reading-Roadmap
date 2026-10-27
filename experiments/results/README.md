# 首轮机制实验实测记录

[实验协议](../README.md) · [首页](../../README.md)

记录日期：2026-10-28（Asia/Shanghai）。这些是本仓库实际运行结果，不是原论文基准指标。

- Python：3.9.6
- 平台：`macOS-27.0.1-arm64-arm-64bit`
- 依赖：仅 Python 标准库；CPU；未使用外部数据。
- 脚本 SHA-256：`8d130aff2172759c00c1379dad62412bcff5c0f11854e5cb79af59f2cc8485d3`
- 语义测试：16 项通过；包括手算值、有限差分、未来信息干预与三个随机种子。

## 运行命令

```bash
python3 experiments/run.py --experiment all --seed 7 --output experiments/results/baseline-seed-7.json
python3 experiments/run.py --experiment all --seed 42 --output experiments/results/baseline-seed-42.json
python3 experiments/run.py --experiment all --seed 2026 --output experiments/results/baseline-seed-2026.json
python3 -m unittest discover -s experiments -p 'test_*.py' -v
```

## 确定性实验（与种子无关）

| 实验 | 观测 | 结论范围 |
|---|---|---|
| residual，深度 20 | 普通输入导数 9.60665e-21；残差输入导数 4.23064 | 小分支配置中直接路径改变梯度传播；不是准确率比较 |
| normalization | BN 第一行变化 1.94357；LN 变化 0 | 训练态统计轴与批次依赖 |
| adam | 初始损失 120.5；最终损失 2.51769e-17 | 已知二次目标上的收敛检查 |
| attention | 因果前缀变化 0；无掩码对照变化 104.407 | 干预未来 K/V 时的信息可见性 |

## 随机实验：独立测试集噪声 MSE

| 种子 | 线性预测 | 零预测基线 | 全部实验检查 | 原始输出 |
|---|---:|---:|---|---|
| 7 | 0.855435 | 1.004107 | 通过 | [JSON](baseline-seed-7.json) |
| 42 | 0.856173 | 1.005330 | 通过 | [JSON](baseline-seed-42.json) |
| 2026 | 0.854786 | 0.994631 | 通过 | [JSON](baseline-seed-2026.json) |

该高斯配置的最优期望噪声 MSE 为 0.857143。有限样本测试值在它附近波动，不能把偶然更低的值解释为超过理论最优。

## 局限

没有运行 ImageNet、机器翻译或图像扩散训练；没有 GPU 性能测量。归一化、注意力与残差只做前向/导数机制检查。Adam 的对照学习率并非搜索后的最佳配置，不能排名算法。diffusion 仅单时刻线性回归；没有完整逆向采样。

CI 的实际运行状态以 GitHub Checks 为准，本地通过不等于远程 CI 已通过。
