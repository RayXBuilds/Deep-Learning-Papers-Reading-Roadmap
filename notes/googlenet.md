# GoogLeNet：怎样让大核分支用得起？

Christian Szegedy、Wei Liu、Yangqing Jia、Pierre Sermanet、Scott Reed、Dragomir Anguelov、Dumitru Erhan、Vincent Vanhoucke、Andrew Rabinovich，*Going Deeper with Convolutions*，2014 年 arXiv:1409.4842，CVPR 2015。核对于 2026-11-21。先修为卷积通道数、乘加次数与拼接。原文训练使用 DistBelief，未确认该次训练的源码发布；BVLC 提供 Caffe 复现。[会议原文](https://openaccess.thecvf.com/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) · [预印本](https://arxiv.org/abs/1409.4842) · [BVLC 实现说明](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet)

## 问题

对同一位置，有的类别线索只涉及局部颜色，有的要看更大的邻域。把不同尺度的响应都交给下一层，可以保留两类信息，但若每个大卷积核都读入全部通道，计算会迅速增长。

## 方法与关键推导

图 2(a) 的朴素模块并行执行多个核和池化，图 2(b) 在昂贵分支前加入 1×1 投影与 ReLU。设空间面积为 A，输入通道 C、输出 K、投影宽度 R。直接 5×5 的乘加数为 25ACK；投影后为 A(CR + 25RK)。只有 R(C + 25K) < 25CK 时此分支才节省乘加。投影层在每个位置学习跨通道组合，给它增加空间视野的是后面的卷积。

表 1 的 3a 模块输入为 28×28×192，5×5 分支先降到 16 通道，再产生 32 个通道。逐位置计算从 153600 次乘加降到 15872 次。四条分支输出 64、128、32、32 个通道，补边保持空间尺寸，拼接得到 28×28×256。池化分支也做投影，以控制下一模块的输入宽度。

## 原论文证据

第 5 节表 1 可逐项核对上面的读数。中间两个辅助分类器在训练阶段提供额外损失，推理时移除。第 7 节的 ILSVRC 2014 分类实验还改变模型数与每图裁剪数，最终竞赛数字包括多模型、多裁剪的平均。它能支持整套设计在比赛设置中的表现，无法直接给出单个降维分支带来的准确率增益。

## 与其他论文的关系

[VGG](vgg.md)用串联层逐渐扩大支持域，这里在分支合并前让不同通道看不同范围。[ResNet](resnet.md)使用逐元素加法，要求捷径和残差形状匹配；Inception 的拼接只要求空间尺寸对齐，通道数可以不同。

## 可验证实验

精读初版，上述为按表手算，未运行 GoogLeNet 专项实验。可固定 A、C、K 扫描 R，对照每层张量的真实乘加与不等式阈值；若再比较速度，应分别计入中间激活读写并报告硬件与批大小。

## 局限与思考

压缩通道会限制送入大核的信息，算术成本无法判断丢失了哪些判别特征。若将节省的预算给另一个分支，需要在总预算相同且评估裁剪数固定时重新比较分类误差。
