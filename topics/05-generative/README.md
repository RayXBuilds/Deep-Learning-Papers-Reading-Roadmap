# 生成模型与图像合成

[全部专题](../README.md) · [首页](../../README.md) · [年份索引](../../docs/roadmaps/timeline.md)

VAE / GAN → WGAN / EBGAN / StyleGAN；DDPM → 潜空间扩散 → Flow Matching；艺术资料作为任务分支阅读。

## 中文精读与阅读顺序

- [VAE：为什么重构项之外，还需要变分下界？](notes/vae.md)
- [VAE 输入跳跃：重构误差下降，生成分布一定更好吗？](notes/vae-input-skip.md)
- [GAN 的判别器如何提供分布学习信号](notes/gan.md)
- [WGAN：分布正在靠近，损失为何仍不变化？](notes/wgan.md)
- [EBGAN 的能量、间隔与样本多样性](notes/ebgan.md)
- [StyleGAN · 姿态与发丝为何能从不同入口控制？](notes/stylegan.md)
- [DDPM：加噪过程如何变成训练信号？](notes/ddpm.md)
- [Latent Diffusion：把扩散移到哪里，细节会在哪一步丢失？](notes/latent-diffusion.md)
- [Flow Matching：样本速度怎样成为生成分布的速度场？](notes/flow-matching.md)
- [Robust Flow Matching：坏的是数据端点，还是条件标签？](notes/robust-flow-matching.md)
- [LipsFlower：输入扰动从哪里进入恢复过程？](notes/lipsflower.md)
- [Flow Matching 的直线性与 Euler 误差：小损失能保证少步采样吗？](notes/flow-straightness.md)
- [DualCast 的角色条件与区域监督](notes/dualcast.md)
- [LLaDA：从整段空白怎样得到一条有依赖的文本序列？](notes/llada.md)

## 原始资料

以下按历史文件名列出全部资料，文件字节保持不变。文件名可能缩写，不能据此推定版本或精读状态；同主题新笔记在上方，逐篇书目校订仍需核对原文。

- [A LEARNED REPRESENTATION FOR ARTISTIC STYLE](<papers/art/A%20LEARNED%20REPRESENTATION%20FOR%20ARTISTIC%20STYLE.pdf>)
- [A Neural Algorithm of Artistic Style](<papers/art/A%20Neural%20Algorithm%20of%20Artistic%20Style.pdf>)
- [Colorful Image Colorization](<papers/art/Colorful%20Image%20Colorization.pdf>)
- [Controlling Perceptual Factors in Neural Style Transfer](<papers/art/Controlling%20Perceptual%20Factors%20in%20Neural%20Style%20Transfer.pdf>)
- [Generative Visual Manipulation on the Natural Image Manifold](<papers/art/Generative%20Visual%20Manipulation%20on%20the%20Natural%20Image%20Manifold.pdf>)
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](<papers/art/Perceptual%20Losses%20for%20Real-Time%20Style%20Transfer%20and%20Super-Resolution.pdf>)
- [Research Blog Inceptionism Going Deeper into Neural Networks](<papers/art/Research%20Blog%20Inceptionism%20Going%20Deeper%20into%20Neural%20Networks.pdf>)
- [Semantic Style Transfer and Turning Two-Bit Doodles into Fine Artwork](<papers/art/Semantic%20Style%20Transfer%20and%20Turning%20Two-Bit%20Doodles%20into%20Fine%20Artwork.pdf>)
- [Texture Networks Feed-forward Synthesis of Textures and Stylized Images](<papers/art/Texture%20Networks%20Feed-forward%20Synthesis%20of%20Textures%20and%20Stylized%20Images.pdf>)
- [Auto-Encoding Variational Bayes](<papers/generative-models/Auto-Encoding%20Variational%20Bayes.pdf>) · [同题精读](notes/vae.md)
- [Building High-level Features](<papers/generative-models/Building%20High-level%20Features.pdf>)
- [Conditional Image Generation with](<papers/generative-models/Conditional%20Image%20Generation%20with.pdf>)
- [DRAW_ A Recurrent Neural Network For Image Generation](<papers/generative-models/DRAW_%20A%20Recurrent%20Neural%20Network%20For%20Image%20Generation.pdf>)
- [Generative Adversarial Nets](<papers/generative-models/Generative%20Adversarial%20Nets.pdf>) · [同题精读](notes/gan.md)
- [Pixel Recurrent Neural Networks](<papers/generative-models/Pixel%20Recurrent%20Neural%20Networks.pdf>)
- [Unsupervised representation learning with deep convolutional generative adversarial networks](<papers/generative-models/Unsupervised%20representation%20learning%20with%20deep%20convolutional%20generative%20adversarial%20networks.pdf>)

## 状态与下一步

DRAW、PixelRNN、DCGAN、风格迁移等历史资料尚未全部完成精读。

[实验指南与实测边界](../../experiments/README.md) · [迁移对照](../../docs/archive/path-migration.md)
