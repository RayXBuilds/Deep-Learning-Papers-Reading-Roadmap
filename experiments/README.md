# 可验证的机制实验

[首页](../README.md) · [实测报告](results/README.md)

环境：Python 3.10+、CPU、仅标准库。无需安装依赖或下载数据。脚本默认固定 seed=7；只有 diffusion 使用随机数，其余输入确定。运行目录为仓库根目录。

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
