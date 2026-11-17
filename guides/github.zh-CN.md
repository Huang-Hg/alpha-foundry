# GitHub 检查与发布边界

[English](github.md) · [贡献说明](../.github/CONTRIBUTING.md)

公开仓库仅跟踪 `foundry/` 核心包、双语用户文档（`guides/`）、经审核的测试数据快照（`data/`）、依赖声明（`pyproject.toml`）以及必要的 GitHub 质量门禁配置（`.github/`、`.pre-commit-config.yaml`）。回归测试（`tests/`）与基准驱动脚本（`scripts/`）仅保留在本地磁盘，不通过改名目录或内嵌源码压缩包上传。C++ 原生扩展构建命令统一维护在根目录 [README](../README.zh-CN.md) 中。

| 门禁类别 | 基础设施 |
| --- | --- |
| 跟踪文件白名单与静态质量 | 路径白名单校验、`pre-commit`、Ruff、`actionlint`、原生扩展编译检查 |
| 依赖与安全扫描 | GitHub Dependency Review、CodeQL（`python`、`c-cpp`、`actions`）、Dependabot |
| 数据快照与基准摘要核验 | `data/SHA256SUMS` 严格校验、`summary.json` 状态核验、Actions Job Summary 展示 |

数值回归与 CPU/CUDA 性能基准在明确授权的云端实例上通过 `pytest`、`pytest-benchmark` 与 `pytest-html` 执行：先完成代码修改并冻结版本，再集中验收；相邻任务复用同一 GPU 会话，每项条件默认测量一次。GitHub Actions 不启动计费 GPU，也不替代四市场规则与数值精度验收。

`guides/benchmarks/` 下仅发布精简双语 Markdown 报告、CSV 表格与 `summary.json`；完整原始日志与归档包统一留在本地 `var/runs/`。已发布报告仅对当时记录的源码摘要与硬件环境负责，不自动背书后续代码变更。

本地静态检查可通过安装 `--group dev` 后执行 `python -m pre_commit run --all-files` 完成。所有 GitHub Actions 与 `pre-commit` 钩子均固定到完整 commit SHA。
