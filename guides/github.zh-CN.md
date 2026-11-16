# GitHub 检查与审核

[English](github.md)

改动在功能分支提交，通过 PR 审核。这里的自动化不会自动合并、发布版本或启动付费 GPU。

| 必须通过的检查 | 检查内容 |
| --- | --- |
| Repository checks | pre-commit 文件与配置检查、Ruff 语法和未定义名称检查、actionlint、PR 依赖审查，以及 CodeQL 分析是否成功完成。 |
| Benchmark evidence | 历史报告和选定数据的校验和；R5 的 JUnit、四市场日线一致性、通过精度资格审查的完整规模测量，以及当前源码的逐文件绑定。 |

CodeQL 扫描 Python、C/C++ 和 Actions 定义，不构建项目。它是安全分析，不能代替 CUDA 编译或数值验证。依赖审查阻止引入高危及严重漏洞。CodeQL 发现的问题显示在仓库 Security 页面；分析成功不代表没有安全问题。

数值验证仍在明确授权的云端执行。计算代码、回归契约、数值依赖约束或测量公式集改变后，按[统一 benchmark 指南](benchmarks/README.zh-CN.md)生成新证据。当前门禁将全部数值源码逐字节绑定到 R5 云端记录。之后的依赖文件合并、安装说明更新单独列在 [source-review.json](benchmarks/timed-l4-20261116/source-review.json)，固定修改前后的摘要，并检查数值依赖约束和 pytest 配置未变。新增文件或进一步修改会阻断门禁，不继承宽泛豁免。

Actions 摘要展示 [R5 L4 报告](benchmarks/timed-l4-20261116/README.zh-CN.md)，保留轻量 FP32 对照失败和旧私有生产基线跳过。门禁核对 508 项通过加这一项明确跳过、smoke 定向修复记录、四市场日线精确一致，以及 4096 数组和 256 程序的完整规模 CPU 正确性审计；不会把失败精度改标通过，也不会把 GPU 主机 CPU 当作速度基线。

成功的证据检查会保存公开验收工件，保留 30 天。验收源码、原始 JUnit、日线对照和数值审计摘要位于 `guides/benchmarks/timed-l4-20261116/`，完整原始压缩包仍保留本地。早期报告保持历史证据及其完整性校验，不再用于证明当前架构已验收。

## 本地静态检查

```bash
python -m pip install --upgrade "pip>=25.1"
python -m pip install --group dev
python -m pre_commit run --all-files --show-diff-on-failure
```

这里只安装隔离的检查工具并检查 Git 跟踪的文件，不导入 Foundry、不编译内核、不执行测试，也不自动改写源码。

Actions 和 pre-commit 工具均固定到不可变的提交。Dependabot 每周为 Actions 和 Python 依赖提出分组更新 PR。pre-commit 工具另用 `python -m pre_commit autoupdate --freeze` 更新，随后执行相同静态检查并审核。依赖更新也不自动合并。

## 仓库设置

`main` 规则要求通过 PR 合并、解决审核讨论，并由 GitHub Actions 针对最新基线通过上述两项检查；禁止删除和强推。当前仓库只有一位维护者，不设置必须由另一人批准的门槛，仍通过草稿 PR 和明确审核控制合并时机。

Actions token 默认只读，仅 CodeQL 获得写入安全分析结果的权限。仓库要求 Actions 使用完整提交 SHA，保留密钥扫描和推送保护，启用 Dependabot 安全告警、修复更新和私密漏洞报告。旧的数值 CI 已停用。

分支规则和安全设置立即生效。Dependabot 定期版本更新、Issue 表单和默认贡献说明需要相应文件进入默认分支后才启用。草稿 PR 可以提前运行新的 PR 检查；保持不合并时，这些默认分支功能仍处于待启用状态。

另见[贡献说明](../.github/CONTRIBUTING.md)与[安全问题报告](../.github/SECURITY.md)。
