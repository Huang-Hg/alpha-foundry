# Independent agent integration: numerical candidates

[简体中文](README.zh-CN.md) · [Integration guide](../../research.md) · [Evidence](evidence.json)

GPT-6.1 Sol independently connected to Foundry using public documentation and runtime discovery. It evaluated six non-formula candidates in two batches: custom numerical factors and a GPU-trained ridge model. Registration, evaluation, feedback-based adaptation, family inspection, retention, release and durable re-evaluation worked. Every candidate failed hard portfolio admission; this run does not establish a profitable strategy or verify commit execution.

## Scope

- Engine source: `28ed260`; one Colab NVIDIA L4, paired arithmetic, disposable portfolio.
- US equities: 491 assets, 148 grid dates including history, 128 evaluation days from 2025-07-01 through 2025-12-31.
- The agent read public instructions, README, research/data guides and runtime schema/signatures. It did not read Foundry implementation, validation examples or existing benchmark reports. Only the public `foundry` import was used.
- The parent prepared the service. The agent authored and ran its own separate RPC client. Factor arithmetic and model fitting used CuPy on L4; file parsing/alignment and NumPy RPC staging used the host. This was not a zero-copy RPC or device-resident feedback benchmark.

## Candidates and feedback

| Round | Candidate | Standalone Sharpe | Hard admission |
| --- | --- | ---: | --- |
| Initial | Volatility-scaled 10-day momentum | -0.586622 | Rejected at G2 |
| Initial | Reversal of that momentum | 1.826576 | Rejected at G2 |
| Initial | Historical-prefix ridge model | 0.194832 | Rejected at G2 |
| Adapted | Volatility-scaled 5-day reversal | 0.939320 | Rejected at G2 |
| Adapted | Tanh-compressed 5-day reversal | 0.935967 | Rejected at G2 |
| Adapted | Ridge with causal score calibration | 0.144209 | Rejected at G2 |

The model fitted 25,301 valid asset/date observations on three causal features. Its latest training label ended at grid row 73; predictions began at row 74. Earlier predictions remained missing. Registration declared the corresponding observation and fitting cutoffs. The adapted ridge used training-prediction standard deviation to calibrate score amplitude. The agent chose the shorter reversal after reviewing the first batch; it did not improve on the first reversal.

This is in-sample research: adapting to evaluation feedback does not constitute an independent holdout. The legacy data lacks publication timestamps and revision vintages; valid client cutoffs cannot repair those source limitations.

## What passed and what remains unverified

All six rows had `ok=true` with `constraints_failed` admission decisions. A two-point `CandidateGrid` completed with zero pass rate; it is too small to support a robustness claim. All six candidates were explicitly retained, their temporary leases released, and durable references re-evaluated. Metrics matched exactly before and after retention; portfolio sources, weights and context remained unchanged. Both clients exited and the L4 runtime was released.

`session.commit` was not invoked. Automatic approval rejected submitting the already-rejected batch because authorization was unclear and publication could be unintended. The agent skipped that step without retrying or bypassing the rejection. Commit execution therefore remains outside this run's verified scope. No elapsed-time instrumentation was collected, so no speed claim is made.

## Onboarding findings

The agent initially passed the dictionary returned by `describe()` to `foundry.export`, causing `TypeError: asdict() should be called on dataclass instances`. It corrected its client to use `json.dumps`; no engine change was needed.

The main documentation gap was signal amplitude. The initial ridge's average exposure was only 0.0387% because scores directly affect allocation; the API does not automatically rank them or fill the investment budget. The bilingual guide now explains that mapping, distinguishes evaluation success from admission, and clarifies dictionary versus typed-report export. People, agents and ML programs can all use the same numerical candidate path.
