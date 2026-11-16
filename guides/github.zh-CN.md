# GitHub 检查与审核

[English](github.md)

改动在功能分支提交，通过 PR 审核。这里的自动化不会自动合并、发布版本或启动付费 GPU。

| 必须通过的检查 | 检查内容 |
| --- | --- |
| Repository checks | pre-commit 文件与配置检查、Ruff 语法和未定义名称检查、actionlint、PR 依赖审查，以及 CodeQL 分析是否成功完成。 |
| Benchmark evidence | 报告和选定数据的校验和、已记录的算子与完整流程对比结果，以及测量源码、报告生成器、公式集与当前版本的一致性。 |

CodeQL 扫描 Python、C/C++ 和 Actions 定义，不构建项目。它是安全分析，不能代替 CUDA 编译或数值验证。依赖审查阻止引入高危及严重漏洞。CodeQL 发现的问题显示在仓库 Security 页面；分析成功不代表没有安全问题。

数值验证仍在明确授权的云端执行。计算代码、验证用例、运行依赖或测量公式集改变后，按[统一 benchmark 指南](benchmarks/README.zh-CN.md)生成新证据。校验允许换行符和文件末尾空白规范化；不会把 AST 相同或旧版本运行成功当作新源码已验证。最终报告可以在测量完成后生成，因此报告生成器另行记录哈希。

Actions 摘要展示耗时、误差、决策结果及报告链接。成功的证据检查还会保存可下载工件，保留 30 天。完整测量包和测量时的源码包继续保存在 `guides/benchmarks/current/`。

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
