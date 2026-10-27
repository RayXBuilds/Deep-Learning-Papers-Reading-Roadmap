# 历史资料入口

[首页](../../README.md) · [历史论文清单](reading-list-2018.md) · [勘误](../errata.md)

原资料区主体整理于 2018 年。旧 PDF 路径保持不变，新精读请从首页进入。以下计数来自本轮开始前的 Git 文件树，PDF 包含书籍、课件和论文，不能等同论文数量。

| 原目录 | PDF 文件数 |
|---|---:|
| [1.0-Book](../../1.0-Book/) | 1 |
| [1.1-Survey](../../1.1-Survey/) | 1 |
| [1.2-Deep_Belief_Network](../../1.2-Deep_Belief_Network/) | 2 |
| [1.3-ImageNet_Evolution](../../1.3-ImageNet_Evolution/) | 4 |
| [1.4-Speech_Recognition_Evolution](../../1.4-Speech_Recognition_Evolution/) | 6 |
| [2.1-Model](../../2.1-Model/) | 8 |
| [2.2-Optimization](../../2.2-Optimization/) | 5 |
| [2.3-Unsupervised_Learning_Deep_Generative_Model](../../2.3-Unsupervised_Learning_Deep_Generative_Model/) | 7 |
| [2.4-RNN_Sequence-to-Sequence_Model](../../2.4-RNN_Sequence-to-Sequence_Model/) | 4 |
| [2.5-Neural_Turing_Machine](../../2.5-Neural_Turing_Machine/) | 5 |
| [2.6-Deep_Reinforcement_Learning](../../2.6-Deep_Reinforcement_Learning/) | 8 |
| [2.7-Deep_Transfer_Learning_Lifelong_Learning_especially_for_RL](../../2.7-Deep_Transfer_Learning_Lifelong_Learning_especially_for_RL/) | 6 |
| [2.8-One_Shot_Deep_Learning](../../2.8-One_Shot_Deep_Learning/) | 2 |
| [3.1-NLP](../../3.1-NLP/) | 9 |
| [3.2-Object_Detection](../../3.2-Object_Detection/) | 4 |
| [3.3-Visual_Tracking](../../3.3-Visual_Tracking/) | 3 |
| [3.4-Image_Caption](../../3.4-Image_Caption/) | 3 |
| [3.5-Machine_Translation](../../3.5-Machine_Translation/) | 6 |
| [3.6-Robotics](../../3.6-Robotics/) | 9 |
| [3.7-Art](../../3.7-Art/) | 9 |
| [3.8-Object_Segmentation](../../3.8-Object_Segmentation/) | 5 |
| [4.0-Other](../../4.0-Other/) | 5 |

共 112 个 PDF。历史 README 列出 130 条资料，其中 18 处标注“暂无”；这些是旧收录状态，不代表外部原文不可用。

新内容只优先维护原文入口、中文解释、关联和实验。完整仓库仍含较大历史资料，若只维护新内容可使用 sparse checkout：

```bash
git clone --filter=blob:none --sparse https://github.com/RayXBuilds/Deep-Learning-Papers-Reading-Roadmap.git
cd Deep-Learning-Papers-Reading-Roadmap
git sparse-checkout set docs notes experiments scripts templates .github
```

稀疏检出时，历史 PDF 在 Git 中仍保留，但本地未下载；点击本地目录链接可能没有文件，可在 GitHub 查看，或将相应旧目录加入 sparse checkout。
