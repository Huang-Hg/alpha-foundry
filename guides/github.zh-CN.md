# GitHub 检查与发布边界

[English](github.md)

仓库保留运行库、双语文档、依赖和必要 GitHub Actions 配置。测试与辅助脚本只保留在本地 `tests/`、`scripts/`，不通过其他目录或压缩包上传。原生构建命令在 README 中。

| 检查 | 使用的基础设施 |
| --- | --- |
| 静态质量 | pre-commit、Ruff、actionlint |
| 依赖与安全 | GitHub Dependency Review、CodeQL、Dependabot |
| 实测报告 | sha256sum 校验选定数据，summary.json 内嵌源码摘要，Actions Job Summary 展示结果，upload-artifact 保存报告 |

数值验收在明确授权的云端，用 pytest、pytest-benchmark、pytest-html。先完成所有修改并冻结源码，再集中验收；相邻任务共用同一 GPU，每条件默认一次。CI 不启动付费 GPU，不代替市场规则或数值正确性检查。

报告只保留精简 Markdown、CSV 表格和内嵌源码摘要的 summary.json；完整原始结果和本地测试代码不进入 Git。旧报告是历史证据，不证明当前源码通过相同验收。

本地静态检查可运行 `python -m pip install --group dev` 后执行 `python -m pre_commit run --all-files`。Actions 和 hooks 固定提交版本。分支保护、安全扫描、依赖审查仍由 GitHub 管理；本轮不推送、不合并，工作流尚未在 GitHub 运行。
