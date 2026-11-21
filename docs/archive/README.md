# 历史资料与目录迁移

[首页](../../README.md) · [统一专题](../../topics/README.md) · [历史论文清单](reading-list-2018.md) · [勘误](../errata.md)

2026-11-21 按用户要求重编排：原有 22 个分类目录及 notes/ 统一迁入 12 个专题。原始资料与新精读不再使用两套顶层分类；请从专题页查看阅读顺序、全部文件和未完成环节。

- [旧路径 → 新位置](path-migration.md)：覆盖 112 个 PDF 与 58 篇已迁移精读；旧地址另提供迁移前固定版本入口。
- [迁移清单及 Git blob 标识](path-migration.json)：内容检查逐个验证 PDF 字节标识，防止丢失或替换。
- 旧 PDF 包含书籍、课件和论文，不能把文件数当作已完成精读数；文件名中的截断和年份仍需逐篇校订。
- 2018 年旧清单的“暂无”是当时的收录状态，不表示今天无法获取。其本地链接已指向新位置，叙述保留历史语境。

## 只下载文本与实验

```bash
git clone --filter=blob:none --sparse https://github.com/RayXBuilds/Deep-Learning-Papers-Reading-Roadmap.git
cd Deep-Learning-Papers-Reading-Roadmap
git sparse-checkout set --no-cone '/*' '!/**/*.pdf'
```

这样保留专题 README 和精读，PDF 仍存在于 Git 文件树。若需要全部原始资料，可运行 `git sparse-checkout disable`，这将下载全部文件。历史版权仍归各自权利人。
