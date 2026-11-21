# MoCo：跨批负例还能用同一把尺子比较吗？

[所属专题与原始资料](../README.md)

Momentum Contrast for Unsupervised Visual Representation Learning

- 作者：Kaiming He、Haoqi Fan、Yuxin Wu、Saining Xie、Ross Girshick。
- 版本：2019-11（v1），2020-03（v3），CVPR 2020。核对于 2026-11-21。[预印本与版本记录](https://arxiv.org/abs/1911.05722) · [会议原文](https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html) · [作者仓库](https://github.com/facebookresearch/moco)
- 先修：[ResNet](resnet.md)、softmax 与指数滑动平均。
- 预印本 DOI：10.48550/arXiv.1911.05722。
- 状态：精读初版，本地机制实验未运行，完整基准未复现。

## 问题

若每次更新只能装下 B 张图，怎样让每个查询仍与上万张图竞争？把旧图的特征留在内存里省下了重算，但模型已更新，昨天的特征和今天的查询可能使用不同的坐标系。

## 方法与关键推导

同一张图的两个增强分别生成单位向量 q 和 k⁺。式 (1) 的单查询损失可改写为 $\log(1+\sum_{i=1}^K\exp((q^Tk_i-q^Tk^+)/T))$。减小一个已很小的负例分数几乎无效，最接近正例的错误匹配贡献较大。先用旧队列算损失，再把当前批的键入队并淘汰最旧一批，避免当前正键又在同一分母中充当负例。

查询编码器接受梯度，键编码器按式 (2) 更新 $\theta_k\leftarrow m\theta_k+(1-m)\theta_q$。对于每步 B 个键、K 能整除 B 的稳态队列，换完一轮需 K/B 步；m 同时控制目标响应新参数的速度。慢更新缓解坐标漂移，不能把历史键变成当前键的精确值。

## 原论文证据

图 3 固定 ResNet-50 与预训练任务，在 ImageNet 冻结表示上线性分类。K=65,536 时，逐样本 memory bank 得到 58.0%，MoCo 得到 60.6%。第 4.1 节固定 K=4,096，只改动量，m=0.9、0.999、0.9999 分别为 55.2%、59.0%、58.9%。第 3.3 节还在多 GPU 训练时打乱键侧批次，防止两视图靠相同设备的 BatchNorm 统计完成匹配；因此字典之外的训练细节也会影响结果。两组对照分别考察字典维护和更新时间尺度。

## 与其他论文的关系

[SimCLR](simclr.md) 为每次查询重新计算批内候选；[BYOL](byol.md) 保留慢目标，却把字典分类改成跨视图回归。读 [CLIP](../../07-multimodal/notes/clip.md) 时再问一次：候选来自哪些样本，正例由谁指定？

## 可验证实验

未运行队列训练。可固定查询权重的一次阶跃变化，依次记录不同 m 下新键的漂移和最旧键的年龄，再分别扩大 B 和 K。

## 局限与思考

更大的字典也会收进更多同类图像。可用标签仅作事后分组，测量同类负例在 softmax 总质量中的占比，再检查线性探针是否同步改善。
