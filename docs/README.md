# 文档与目录导航

[返回项目首页](../README.md)

## 学习内容

| 入口 | 内容 |
|---|---|
| [阅读方法与核心顺序](roadmaps/README.md) | 先修知识、阅读方法和核心学习单元 |
| [训练与视觉](roadmaps/training-and-vision.md) | 卷积网络、残差、归一化和优化 |
| [序列与注意力](roadmaps/sequence-and-attention.md) | 序列模型、注意力及 Transformer 的扩展方向 |
| [生成模型](roadmaps/generative-models.md) | VAE、GAN 与 DDPM 的目标对照 |
| [论文索引](papers.md) | 原文来源、阅读问题与精读状态 |
| [技术关联](connections.md) | 论文间关系、演进图与比较方法 |
| [实验指南](../experiments/README.md) | 环境、命令、对照、验收和失败诊断 |
| [实测记录](../experiments/results/README.md) | 数值结果、环境与机器可读输出 |
| [历史资料](archive/README.md) | 原分类目录、旧清单和稀疏下载方法 |

## 仓库结构

```text
docs/roadmaps/          学习路线与验收目标
notes/                  中文论文精读
experiments/            机制实验、测试与结果
scripts/                本地内容检查
templates/              精读与实验报告模板
docs/archive/           历史清单和原目录索引
1.0-Book/ … 4.0-Other/   原有 PDF 资料，保持路径稳定
```

## 维护内容

- [贡献规范与检查命令](../CONTRIBUTING.md)
- [选题队列与完成条件](maintenance.md)
- [精读模板](../templates/paper-note.md)
- [实验报告模板](../templates/experiment-report.md)
- [已核实勘误](errata.md)

稀疏检出时，Git 中保留的 PDF 可能没有下载到本地；需要时按历史资料页说明添加相应目录。
