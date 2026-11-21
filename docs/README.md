# 文档与目录导航

[返回项目首页](../README.md)

## 学习内容

| 入口 | 内容 |
|---|---|
| [阅读方法与核心顺序](roadmaps/README.md) | 先修知识、阅读方法和核心学习单元 |
| [训练与视觉](roadmaps/training-and-vision.md) | 卷积网络、残差、归一化和优化 |
| [序列与注意力](roadmaps/sequence-and-attention.md) | 序列模型、注意力及 Transformer 的扩展方向 |
| [生成模型](roadmaps/generative-models.md) | VAE、GAN、DDPM 与 Flow Matching 的目标对照 |
| [论文索引](papers.md) | 原文来源、阅读问题与精读状态 |
| [技术关联](connections.md) | 论文间关系、演进图与比较方法 |
| [实验指南](../experiments/README.md) | 环境、命令、对照、验收和失败诊断 |
| [实测记录](../experiments/results/README.md) | 数值结果、环境与机器可读输出 |
| [每日维护记录](daily/README.md) | 当日新增和实质修订的内容与验证记录 |
| [专题总览](../topics/README.md) | 统一阅读顺序、精读与原始资料 |
| [迁移与历史说明](archive/README.md) | 新旧路径对照、旧清单与稀疏下载 |

## 仓库结构

```text
topics/                 统一专题入口
  <topic>/README.md     阅读顺序、覆盖状态与资料清单
  <topic>/notes/        中文论文精读
  <topic>/papers/       原始资料，按子方向归档
docs/roadmaps/          跨专题路线与年份索引
experiments/            机制实验、测试与结果
scripts/                内容、链接与迁移完整性检查
templates/              精读与实验报告模板
docs/archive/           旧清单、迁移对照与历史说明
```

## 维护内容

- [贡献规范与检查命令](../CONTRIBUTING.md)
- [选题队列与完成条件](maintenance.md)
- [精读模板](../templates/paper-note.md)
- [实验报告模板](../templates/experiment-report.md)
- [已核实勘误](errata.md)

稀疏检出时，Git 中保留的 PDF 可能没有下载到本地；需要时按历史资料页说明添加相应目录。
