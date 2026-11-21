# 论文索引与阅读状态

[首页](../README.md) · [路线](roadmaps/README.md) · [历史清单](archive/reading-list-2018.md)

“精读初版”表示已有中文问题分析、方法、原文证据位置、局限和关联；实验是否已验证另行标注，不表示完整复现。“原文入口”表示只完成选题和来源整理，精读与实验尚未完成。年份以首次预印本为主，必要时另列会议年份。此表是新路线的精选索引，不替代历史全部条目。

| 论文 / 年份 | 学习问题 | 一手来源 | 本仓库状态 |
|---|---|---|---|
| MoCo / 2019，CVPR 2020 | 跨批负例还能用同一把尺子比较吗？ | [原文](https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html) | [精读初版](../notes/moco.md)，本地实验未运行 |
| SimCLR / 2020，ICML 2020 | 为什么训练时用投影头，评估时读它的输入？ | [原文](https://proceedings.mlr.press/v119/chen20j.html) | [精读初版](../notes/simclr.md)，候选集机制已验证 |
| BYOL / 2020，NeurIPS 2020 | 没有负例时，在线网络在追谁？ | [原文](https://proceedings.neurips.cc/paper_files/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html) | [精读初版](../notes/byol.md)，本地实验未运行 |
| GPT-3 / 2020，NeurIPS 2020 | 示例怎样在参数固定时改变回答？ | [原文](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html) | [精读初版](../notes/gpt-3.md)，本地实验未运行 |
| Swin Transformer / 2021，ICCV 2021 | 移动半个窗口后，哪些图像块第一次能交流？ | [原文](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html) | [精读初版](../notes/swin.md)，本地实验未运行 |
| Masked Autoencoders（MAE） / 2021，CVPR 2022 | 解码器为什么要拿回原始块位置？ | [原文](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html) | [精读初版](../notes/mae.md)，掩码重建机制已验证 |
| AlexNet / 2012 | 局部共享与整图分类 | [原文](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) | [精读初版](../notes/alexnet.md)，专项实验未运行 |
| word2vec（CBOW / Skip-gram） / 2013，ICLR Workshop | 上下文与中心词互换后，监督如何改变？ | [原文](https://arxiv.org/abs/1301.3781) | [精读初版](../notes/word2vec.md)，专项实验未运行 |
| Dropout / 2014，JMLR | 哪些输出可以用期望缩放？ | [原文](https://jmlr.org/papers/v15/srivastava14a.html) | [精读初版](../notes/dropout.md)，掩码枚举已验证 |
| VGG / 2014，ICLR 2015 | 小核如何组合空间证据？ | [原文](https://arxiv.org/abs/1409.1556) | [精读初版](../notes/vgg.md)，脉冲响应已验证 |
| GoogLeNet / 2014，CVPR 2015 | 通道投影怎样控制多尺度分支成本？ | [原文](https://arxiv.org/abs/1409.4842) | [精读初版](../notes/googlenet.md)，专项实验未运行 |
| Seq2Seq / 2014，NIPS | 固定摘要与源句反转改变哪些依赖？ | [原文](https://arxiv.org/abs/1409.3215) | [精读初版](../notes/seq2seq.md)，专项实验未运行 |
| U-Net / 2015，MICCAI 2015 | 跳接裁剪与相邻细胞边界 | [原文](https://arxiv.org/abs/1505.04597) | [精读初版](../notes/u-net.md)，分割实验未运行 |
| Faster R-CNN / 2015，NIPS 2015 | 共享特征上的候选框生成 | [原文](https://arxiv.org/abs/1506.01497) | [精读初版](../notes/faster-rcnn.md)，RPN 未验证 |
| SSD / 2015，ECCV 2016 | 直接分类、多层铺框与难负例 | [原文](https://arxiv.org/abs/1512.02325) | [精读初版](../notes/ssd.md)，检测实验未运行 |
| DenseNet / 2016，CVPR 2017 | 块内拼接、增长率与特征复用 | [原文](https://arxiv.org/abs/1608.06993) | [精读初版](../notes/densenet.md)，通道消融未运行 |
| FPN / 2016，CVPR 2017 | 金字塔融合与区域选层 | [原文](https://arxiv.org/abs/1612.03144) | [精读初版](../notes/fpn.md)，RoI 路由已验证 |
| MobileNet v1 / 2017，arXiv 技术报告 | 空间滤波与逐点混合 | [原文](https://arxiv.org/abs/1704.04861) | [精读初版](../notes/mobilenet-v1.md)，线性卷积核分解已验证 |
| ResNet / 2015，CVPR 2016 | 深度增加后的训练退化 | [原文](https://arxiv.org/abs/1512.03385) | [精读初版](../notes/resnet.md)；residual 与 residual_gate.py（加法后激活/抵消补充） |
| BatchNorm / 2015 | 跨样本统计与训练/推理 | [原文](https://arxiv.org/abs/1502.03167) | [精读初版](../notes/batch-normalization.md)；normalization 与 bn_jacobian.py（统计量求导/冻结对照） |
| LayerNorm / 2016 | 单样本内部统计 | [原文](https://arxiv.org/abs/1607.06450) | [精读初版](../notes/layer-normalization.md)；normalization；含统计轴/仿射输出手算对照 |
| Adam / 2014，ICLR 2015 | 自适应矩估计与偏差校正 | [原文](https://arxiv.org/abs/1412.6980) | [精读初版](../notes/adam.md)；adam |
| Bahdanau Attention / 2014，ICLR 2015 | 动态读取源序列 | [原文](https://arxiv.org/abs/1409.0473) | [精读初版](../notes/bahdanau-attention.md)；additive_attention.py |
| Transformer / 2017 | 用注意力组织序列计算 | [原文](https://arxiv.org/abs/1706.03762) | [精读初版](../notes/transformer.md)；attention |
| VAE / 2013，ICLR 2014 | 近似后验、变分目标与重参数化 | [原文](https://arxiv.org/abs/1312.6114) | [精读初版](../notes/vae.md)；vae_objectives.py |
| VAE 输入跳跃 / 2026 | 重构坐标与生成评价 | [原文](https://arxiv.org/abs/2610.37456) | [精读初版](../notes/vae-input-skip.md)；共享 VAE 机制实验；预印本评审状态未独立核实 |
| GAN / 2014，NIPS 2014 | 对抗式生成目标与梯度 | [原文](https://arxiv.org/abs/1406.2661) | [精读初版](../notes/gan.md)，gan_objective.py，神经网络训练待实现 |
| WGAN / 2017，ICML 2017 | 运输距离与 critic 约束 | [原文](https://arxiv.org/abs/1701.07875) | [精读初版](../notes/wgan.md)，wasserstein_geometry.py，神经网络训练待实现 |
| EBGAN / 2016，ICLR 2017 | 自编码能量、间隔与表示方向 | [原文](https://arxiv.org/abs/1609.03126) | [精读初版](../notes/ebgan.md)，ebgan_energy.py 与 ebgan_dimension.py，含 PT 维数下界，神经网络训练待实现 |
| DDPM / 2020 | 加噪与噪声预测 | [原文](https://arxiv.org/abs/2006.11239) | [精读初版](../notes/ddpm.md)；diffusion |
| Flow Matching / 2022，ICLR 2023 | 条件速度、边缘场与时间方向 | [原文](https://arxiv.org/abs/2210.02747) | [精读初版](../notes/flow-matching.md)，flow_matching.py 与 flow_mixture.py，高斯/双峰机制已验证，图像训练未复现 |
| Robust Flow Matching / 2026 | 端点污染与条件标签噪声 | [原文](https://arxiv.org/abs/2611.13324) | [精读初版](../notes/robust-flow-matching.md)，robust_flow.py 与 trs_buffer.py（评分缓冲区）；作者标注 NeurIPS 已接收，会议记录未独立核对；训练未复现 |
| LipsFlower / 2026 | 初值与条件扰动的传播 | [原文](https://arxiv.org/abs/2611.13995v1) | [精读初版](../notes/lipsflower.md)，restoration_stability.py；未同行评审预印本，官方代码未核实，图像训练未复现 |
| Flow 直线性与 Euler 误差 / 2026 | 耦合统计与数值终点误差 | [原文](https://arxiv.org/abs/2611.18343v1) | [精读初版](../notes/flow-straightness.md)，euler_error.py 与 potential_gaussian.py（势流补充）；未同行评审，官方代码未核实，匹配耦合未复现 |
| DualCast / 2026 | 双参考条件与区域监督 | [原文](https://arxiv.org/abs/2611.05825) | [精读初版](../notes/dualcast.md)，预印本，dualcast_regions.py，视频训练未复现 |
| AdamW / 2017，ICLR 2019 | 解耦权重衰减 | [原文](https://arxiv.org/abs/1711.05101) | [精读初版](../notes/adamw.md)；adamw_decay.py |
| BERT / 2018，NAACL 2019 | 双向 Transformer 预训练 | [原文](https://arxiv.org/abs/1810.04805) | [精读初版](../notes/bert.md)；attention 与 bert_mlm.py；输入扰动和 MLM/NSP 训练未运行 |
| ViT / 2020，ICLR 2021 | 图像块序列的 Transformer | [原文](https://arxiv.org/abs/2010.11929) | [精读初版](../notes/vit.md)；vit_patches.py；分类训练未复现 |
| CLIP / 2021，ICML 2021 | 图文配对监督与对比学习 | [原文](https://arxiv.org/abs/2103.00020) | [精读初版](../notes/clip.md)；clip_objective.py；预训练及零样本分类未复现 |
| LoRA / 2021，ICLR 2022 | 冻结权重与低秩适配 | [原文](https://arxiv.org/abs/2106.09685) | [精读初版](../notes/lora.md)；lora_mechanics.py；任务微调未复现 |
| ELMo / 2018，NAACL-HLT | 同一个词怎样获得随句子变化的表示？ | [原文](https://aclanthology.org/N18-1202.pdf) | [精读初版](../notes/elmo.md)，未运行 biLM |
| GPT-1 / 2018，OpenAI 技术报告 | 只读左侧的模型怎样完成句对分类？ | [原文](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | [精读初版](../notes/gpt-1.md)，未运行微调 |
| Transformer-XL / 2019，ACL | 分段计算怎样保留上段的语言线索？ | [原文](https://aclanthology.org/P19-1285.pdf) | [精读初版](../notes/transformer-xl.md)，未运行段级递归实验 |
| XLNet / 2019，NeurIPS | 打乱预测顺序，怎样保留词序又防止偷看答案？ | [原文](https://papers.neurips.cc/paper/8812-xlnet-generalized-autoregressive-pretraining-for-language-understanding.pdf) | [精读初版](../notes/xlnet.md)，已验证排列可见性 |
| T5 / 2019，JMLR 2020 | 一段缺失文本如何变成可生成的标签？ | [原文](https://www.jmlr.org/papers/volume21/20-074/20-074.pdf) | [精读初版](../notes/t5.md)，已验证片段去噪序列 |
| StyleGAN / 2018，CVPR 2019 | 姿态与发丝为何能从不同入口控制？ | [原文](https://arxiv.org/html/1812.04948) | [精读初版](../notes/stylegan.md)，未运行生成器 |
| Latent Diffusion / 2021，CVPR 2022 | 压缩倍率与潜空间去噪 | [原文](https://arxiv.org/abs/2112.10752) | [精读](../notes/latent-diffusion.md)，模型未运行 |
| Chinchilla / 2022 | 等 FLOPs 预算中的参数与 token | [原文](https://arxiv.org/abs/2203.15556) | [精读](../notes/chinchilla.md)，预训练未运行 |
| FlashAttention / 2022 | 跨块 softmax 与精确注意力 | [原文](https://arxiv.org/abs/2205.14135) | [精读](../notes/flashattention.md)，行归约已验证 |
| InstructGPT / 2022 | 示范、成对奖励与 KL 正则 | [原文](https://arxiv.org/abs/2203.02155) | [精读](../notes/instructgpt.md)，PPO 未运行 |
| DPO / 2023 | 由策略相对参考分布的赔率还原偏好 | [原文](https://arxiv.org/abs/2305.18290) | [精读](../notes/dpo.md)，概率恒等式已验证 |
| QLoRA / 2023 | 块量化基座与低秩适配梯度 | [原文](https://arxiv.org/abs/2305.14314) | [精读](../notes/qlora.md)，量化微调未运行 |

本表最近维护：2026-11-21；各论文的来源核对日期见精读元数据。新增内容优先引用公开一手来源，不新增论文 PDF 镜像。按首次公开年份浏览[精读时间线](roadmaps/timeline.md)。
