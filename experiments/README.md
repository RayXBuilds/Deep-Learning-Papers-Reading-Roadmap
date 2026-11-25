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

[中文精读](../topics/04-language/notes/bahdanau-attention.md) · [协议与实测](results/additive-attention.md)

```bash
python3 experiments/additive_attention.py --output /tmp/additive-attention.json
```

这是第六项独立机制实验，不包含在 `run.py --experiment all` 的核心五项中。固定人工输入，检查 query 依赖、去掉非线性的消融、padding 干预及变长序列一致性。无随机采样或训练；输出单独携带脚本哈希，语义测试由统一 discover 命令运行。没有修改或重生成原有基线。

## VAE 目标与生成评价专题

[经典精读](../topics/05-generative/notes/vae.md) · [输入跳跃精读](../topics/05-generative/notes/vae-input-skip.md) · [协议与实测](results/vae-objectives.md)

```bash
python3 experiments/vae_objectives.py --output /tmp/vae-objectives.json
```

第七项独立机制实验，两个精读共享，不包含在核心 `run.py --experiment all` 中。标准库 CPU：固定线性高斯模型，种子 7 / 42 / 2026 各抽样 100,000 次，检查 ELBO、路径梯度、归一化密度与重构坐标。它也使用随机数，但不训练参数；不验证图像生成或 DART 数据表现。验收条件和负对照见实测协议。

## AdamW 权重衰减专题

[中文精读](../topics/02-training/notes/adamw.md) · [协议与实测](results/adamw-decay.md)

```bash
python3 experiments/adamw_decay.py --output /tmp/adamw-decay.json
```

第八项独立机制实验，不包含在 `run.py --experiment all` 中。固定输入、无随机数和第三方依赖；比较首步手算、梯度历史、衰减系数约定与同预算二次目标轨迹。完整逐步记录写入 `--output` 指定的 JSON；终端只输出检查与终值。没有图像训练或泛化排名。

## LoRA 初始化与合并专题

[中文精读](../topics/10-transfer/notes/lora.md) · [协议与实测](results/lora-mechanics.md)

```bash
python3 experiments/lora_mechanics.py --output /tmp/lora-mechanics.json
```

第九项独立机制实验，不包含在 `run.py --experiment all` 中。CPU 标准库：手工矩阵、有限差分与种子 7 / 42 / 2026 的初始化，两步 SGD 追踪梯度；含双零初始化、固定 dropout 和重复增量负对照。没有语言模型、数据集或显存测量；小矩阵主配置不用于证明参数效率。

## ViT 图像块与位置专题

[中文精读](../topics/03-vision/notes/vit.md) · [协议与实测](results/vit-patches.md)

```bash
python3 experiments/vit_patches.py --output /tmp/vit-patches.json
```

第十项独立机制实验，不包含在核心 `run.py --experiment all` 中。固定人工图像、无随机数，以独立卷积循环与手算验证块投影，并比较有/无位置标签时的 token 排列干预。CPU 标准库，无训练、权重下载或图像分类指标。

## BERT MLM 损失专题

[精读](../topics/04-language/notes/bert.md) · [协议与实测](results/bert-mlm.md)

```bash
python3 experiments/bert_mlm.py --output /tmp/bert-mlm.json
```

第十一项独立机制实验，不包含在核心 `run.py --experiment all` 中。固定 logits、标签与权重，验证选中未替换位置和 padding 的损失语义；CPU 标准库，无随机数、输入扰动采样或预训练。

## CLIP 配对目标专题

[精读](../topics/07-multimodal/notes/clip.md) · [协议与实测](results/clip-objective.md)

```bash
python3 experiments/clip_objective.py --output /tmp/clip-objective.json
```

第十二项独立机制实验，不包含在核心 `run.py --experiment all` 中。固定二维向量、CPU 标准库、无随机数；检查双向损失、配对置换、重复表示、归一化与候选概率。没有编码器训练、官方权重下载或零样本准确率。

## GAN 目标专题

运行 `python3 experiments/gan_objective.py --output /tmp/gan-objective.json`，检查有限概率表上的最优判别器、JS 等式与两种生成器损失的梯度。CPU 标准库，输入确定，seed=null。该专题单独运行，神经网络训练待实现。[精读](../topics/05-generative/notes/gan.md) · [协议与实测](results/gan-objective.md)

## WGAN 距离专题

运行 `python3 experiments/wasserstein_geometry.py --output /tmp/wasserstein-geometry.json`，检查一维等权经验分布的运输代价、JS 和 critic 约束。CPU 标准库，输入确定，seed=null。该专题单独运行，神经网络训练待实现。[精读](../topics/05-generative/notes/wgan.md) · [实测](results/wasserstein-geometry.md)

## EBGAN 能量专题

[精读](../topics/05-generative/notes/ebgan.md) · [实测](results/ebgan-energy.md)

```bash
python3 experiments/ebgan_energy.py --output /tmp/ebgan-energy.json
```

第十五项机制实验，独立于核心五项。CPU 标准库、人工输入、seed=null，检查有限能量表的最优响应、间隔导数与 pulling-away 几何。自编码器及生成器训练未运行。

2026-11-08 增加同一专题的维数补充，运行 `python3 experiments/ebgan_dimension.py`，检查 PT 下界与达到下界的重复表示。[配置与结果](results/ebgan-dimension.md)

## DualCast 区域监督专题

[精读](../topics/05-generative/notes/dualcast.md) · [实测](results/dualcast-regions.md)

运行 `python3 experiments/dualcast_regions.py`。第十六项机制实验，CPU 标准库，固定掩码和残差，检查重叠权重、条件丢弃与损失归一化。

## Flow Matching 速度场专题

[精读](../topics/05-generative/notes/flow-matching.md) · [实测](results/flow-matching.md)

运行 `python3 experiments/flow_matching.py`。第十七项机制实验，独立于核心五项。CPU 标准库，三个种子的高斯端点回归与确定性解析流对照，检查总体损失常数差、积分误差和时间反转符号。没有神经网络训练或图像基准。

2026-11-13 补充同一 Flow Matching 专题的双峰对照：`python3 experiments/flow_mixture.py`。独立训练/测试数据、三个种子、解析场及总体线性场同预算积分；[协议与结果](results/flow-mixture.md)。专题总数仍为 17。

## Robust Flow Matching 监督可靠性专题

运行 `python3 experiments/robust_flow.py`。第十八项机制实验，固定仿射场与三类损失表，CPU 标准库，无随机数。检查残差分解、抵消反例、CCS 运算顺序及时间采样；未训练鲁棒生成器。[精读](../topics/05-generative/notes/robust-flow-matching.md) · [实测](results/robust-flow.md)

2026-11-15 补充同一 Robust Flow Matching 专题：`python3 experiments/trs_buffer.py` 验证样本索引、EMA、暖启动与已访问分数的分位数。[实测](results/trs-buffer.md)。专题数量不变。

## LipsFlower 输入敏感性专题

运行 `python3 experiments/restoration_stability.py`。第十九项机制实验，CPU 标准库、确定性标量 ODE、seed=null；验证初值与条件增益、配对随机性及积分误差，不含神经网络训练。[精读](../topics/05-generative/notes/lipsflower.md) · [协议与实测](results/restoration-stability.md)

## Euler 主误差专题

第二十项机制实验：`python3 experiments/euler_error.py`。确定性 CPU 标量场，检查主系数、误差阶和传播项负对照。[精读](../topics/05-generative/notes/flow-straightness.md) · [实测](results/euler-error.md)。未构造匹配直线性的 FM 耦合。

2026-11-18 补充同一 Flow 直线性专题：`python3 experiments/potential_gaussian.py` 比较二维高斯场的旋度、环流和对称化负对照。[实测](results/potential-gaussian.md)。专题总数仍为 20。

## ResNet 激活与分支抵消补充

`python3 experiments/residual_gate.py`：固定仿射分支、输入和深度，验证加法后 ReLU、分支抵消及深度累乘。[精读](../topics/03-vision/notes/resnet.md) · [实测](results/residual-gate.md)。属于已有 residual 专题，专题总数仍为 20，不是网络训练。

## BatchNorm 反向传播补充

`python3 experiments/bn_jacobian.py`：单特征批的解析导数对照有限差分，检查公共平移、有限 ε 的径向导数和冻结统计量。[精读](../topics/02-training/notes/batch-normalization.md) · [实测](results/bn-jacobian.md)。属于既有 normalization 专题，仍为 20 个机制专题；没有网络训练或滑动统计估计。

## Dropout 与 VGG

运行 `python3 experiments/early_mechanisms.py` 检查掩码期望和卷积脉冲响应。[协议、实测及边界](results/early-mechanisms.md)

## MobileNet v1 与 FPN

运行 `python3 experiments/vision_operators.py`，比较分解卷积的两条线性计算路径，再将不同尺度的框分到 P2–P5，与固定 P4 对照。[配置、阈值与实测结果](results/vision-operators.md)

## 排列可见性与片段标签

运行 `python3 experiments/pretraining_layouts.py` 穷举 XLNet 的四位置查询和 T5 的六词缺段，比较内容流自读及全文重建基线。[配置、结果与边界](results/pretraining-layouts.md)

## 视觉自监督目标

- `python3 experiments/view_candidates.py` 比较 SimCLR 正确分母、误含自身和重复方向负例。[协议与结果](results/view-candidates.md)
- `python3 experiments/masked_patches.py` 比较 MAE 遮挡误差和全图误差，核对解码器的块顺序。[协议与结果](results/masked-patches.md)

## 分块注意力与偏好概率

| 运行命令（标准库，CPU） | 输入与基线 | 验收与解释范围 |
|---|---|---|
| `python3 experiments/attention_tiles.py` | 17 个带 +1000 偏移的分数及标量值，种子 7、42、2026，块宽 1、4、8、17，对照整行稳定 softmax | 最大绝对误差 < 1e-12；在两块质量悬殊的反例上，等权平均局部输出误差 > 0.49。只验证逐行前向，不测 GPU。[记录](results/attention-tiles.json) |
| `python3 experiments/preference_odds.py` | 三回答、非均匀参考分布，枚举 β=0.25、1、4 及三对候选。确定性，无随机种子 | 由策略还原的偏好概率与奖励差基线的最大误差 < 1e-12；遗漏参考项的最大误差 > 0.5。未训练语言模型。[记录](results/preference-odds.json) |

[SSD 递推与 LLaDA 掩码权重](results/ssd-and-mask.md) 给出两项 CPU 实验的输入、独立参照、命令及验收阈值。

## 指针网络的位置读出

运行 `python3 experiments/pointer_readout.py`。六种顺序与扩展到四个候选后的概率须符合整数比，两组权重的读出均值须相同，目标位置损失分别为 ln(4/3)、ln(4)。绝对误差容限 1e−12。[精读与输入](../topics/08-memory/notes/pointer-networks.md) · [CPU 实测与脚本哈希](results/pointer-readout.json)

## 多分辨率响应的连续定位

运行 `python3 experiments/ccot_localization.py`。用解析峰位检查网格定位误差与牛顿细化误差，并核对混叠信号在网格内外的差别。[输入、阈值与边界](../topics/11-tracking/notes/c-cot.md) · [实测数据](results/ccot-localization.json)

## Dueling 输出头

运行 `python3 experiments/dueling_head.py`。四个动作中只监督一个，将两个输出头在所选动作上都拟合到 1，再检查其余动作的误差怎样随真值方向改变。[推导与验收条件](../topics/09-rl-robotics/notes/dueling-networks.md) · [实测记录](results/dueling-head.json)

## 蒸馏温度与软目标梯度

运行 `python3 experiments/distillation_temperature.py`，比较首选相同但非首选类排序相反的教师，并与中心化得分差的高温极限核对。[推导与阈值](../topics/10-transfer/notes/knowledge-distillation.md) · [实测](results/distillation-temperature.json)
