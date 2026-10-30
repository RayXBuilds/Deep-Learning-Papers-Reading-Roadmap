# 可验证的机制实验

[首页](../README.md) · [实测报告](results/README.md)

环境：Python 3.10+、CPU、仅标准库。无需安装依赖或下载数据。核心 `run.py` 默认固定 seed=7；核心五项中只有 diffusion 使用随机数，其余输入确定。独立专题的种子另见下文。运行目录为仓库根目录。

```bash
python3 experiments/run.py --experiment all --seed 7 --output /tmp/roadmap-results.json
python3 experiments/run.py --experiment attention
python3 -m unittest discover -s experiments -p 'test_*.py' -v
```

## 实验协议

| 实验 | 假设/对照 | 关键验收 | 不验证什么 |
|---|---|---|---|
| residual | 相同小 tanh 分支，有/无恒等捷径 | 深度 1/5/20 的解析与差分导数相对误差 < 1e-6；默认深度 20 普通导数 < 1e-18、残差导数 > 1 | 不训练权重，不比较图像准确率 |
| normalization | 固定第一行，仅修改同批另一行 | LN 第一行变化 < 1e-12，训练态 BN 变化 > 0.1；对应统计轴均值接近零 | 没有推理态滑动统计、反向传播或训练速度结论 |
| adam | 二维二次函数，三种更新规则 | 全部终值有限；默认 Adam 最终损失 < 1e-8；第一步另有手算测试 | 没有调参排名、随机噪声或泛化比较 |
| attention | 修改未来 K、V，有/无因果掩码 | 权重行和为 1、未来权重为 0、因果前缀不变、无掩码对照改变 | 没有多头、位置编码、完整模型或翻译训练 |
| diffusion | 一维高斯、固定时刻，独立抽样训练/测试 | 均值误差 < 0.08、方差误差 < 0.15、线性系数误差 < 0.03、测试 MSE 比零预测低至少 0.05 | 没有神经网络、完整逆向链或图像生成 |

随机容忍值是此教学配置的验收界限，不是显著性检验或对所有随机种子的保证。测试额外覆盖 7、42、2026 三个种子。

## 输出与失败诊断

JSON 含配置、数值、逐项检查、总状态、种子、Python/平台及脚本 SHA-256。失败返回退出码 1；参数错误返回 argparse 错误。不要只看退出状态，还应阅读指标是否支持所写结论。

- residual 失败：检查有限差分步长、链式法则和分支尺度。极小导数用相对误差核验。
- normalization 失败：先查统计轴，再确认比较的是训练态 BN。
- attention 失败：检查是否在 softmax 之前屏蔽未来，且 softmax 沿 key 轴归一化。
- adam 失败：检查 step 是否从 1 开始，是否对 m、v 都作偏差校正。
- diffusion 失败：先查训练/测试是否重用数据，再查 α_bar、方差与噪声定义；更改参数后重新解释容忍值。

## 扩展实验

参数位于 [run.py](run.py) 各实验函数中。改代码后用新文件名保存报告，记录假设和变化，不覆盖基线冒充原结果。新增训练实验需要采用 [报告模板](../templates/experiment-report.md)，固定数据版本、划分和预算。

## 加性注意力专题

[中文精读](../notes/bahdanau-attention.md) · [协议与实测](results/additive-attention.md)

```bash
python3 experiments/additive_attention.py --output /tmp/additive-attention.json
```

这是第六项独立机制实验，不包含在 `run.py --experiment all` 的核心五项中。固定人工输入，检查 query 依赖、去掉非线性的消融、padding 干预及变长序列一致性。无随机采样或训练；输出单独携带脚本哈希，语义测试由统一 discover 命令运行。没有修改或重生成原有基线。

## VAE 目标与生成评价专题

[经典精读](../notes/vae.md) · [输入跳跃精读](../notes/vae-input-skip.md) · [协议与实测](results/vae-objectives.md)

```bash
python3 experiments/vae_objectives.py --output /tmp/vae-objectives.json
```

第七项独立机制实验，两个精读共享，不包含在核心 `run.py --experiment all` 中。标准库 CPU：固定线性高斯模型，种子 7 / 42 / 2026 各抽样 100,000 次，检查 ELBO、路径梯度、归一化密度与重构坐标。它也使用随机数，但不训练参数；不验证图像生成或 DART 数据表现。验收条件和负对照见实测协议。

## AdamW 权重衰减专题

[中文精读](../notes/adamw.md) · [协议与实测](results/adamw-decay.md)

```bash
python3 experiments/adamw_decay.py --output /tmp/adamw-decay.json
```

第八项独立机制实验，不包含在 `run.py --experiment all` 中。固定输入、无随机数和第三方依赖；比较首步手算、梯度历史、衰减系数约定与同预算二次目标轨迹。完整逐步记录写入 `--output` 指定的 JSON；终端只输出检查与终值。没有图像训练或泛化排名。
