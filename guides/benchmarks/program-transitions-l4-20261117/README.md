# L4: consecutive admissions and portfolio reallocation

[简体中文](README.zh-CN.md) · [Summary JSON](summary.json)

Measured on a single NVIDIA L4 (one run per before/after condition). All three workflows mutate portfolio state—covering both consecutive candidate admissions and explicit two-member weight reallocation.

| Workload (registered / evaluated) | Research before → after (s) | Speedup | Lifecycle before → after (s) | New admissions | Weight transition |
| --- | ---: | ---: | ---: | ---: | --- |
| Admissions / shared (`4 / 4`) | 82.735 → 55.700 | 1.49× | 110.121 → 57.554 | 2 | `[1.0] → [2.0]` |
| Reallocation / shared (`4 / 2`) | 55.931 → 37.582 | 1.49× | 62.149 → 43.767 | 0 | `[0.25, 0.25] → [0.5, 0.5]` |
| Reallocation / independent (`4 / 2`) | 82.768 → 55.302 | 1.50× | 91.586 → 63.898 | 0 | `[0.25, 0.25] → [0.5, 0.5]` |
| Admissions / shared (`16 / 16`) | 70.818 → 55.568 | 1.27× | 72.659 → 57.371 | 2 | `[1.0] → [2.0]` |
| Reallocation / shared (`16 / 14`) | 56.422 → 37.464 | 1.51× | 62.629 → 43.638 | 0 | `[0.25, 0.25] → [0.5, 0.5]` |
| Reallocation / independent (`16 / 14`) | 83.046 → 55.380 | 1.50× | 91.605 → 63.958 | 0 | `[0.25, 0.25] → [0.5, 0.5]` |

## Numerical parity and scope

- **Exact parity:** candidate-return max absolute error **0**, metric max scaled error **0**, admission decision differences **0**. Weight feasibility, selected weight vectors, commit results, and pre/post-allocation portfolio returns match identically.
- **State transitions:** in the admissions workload, the first two programs consecutively replace/increase exposure to rising assets while trailing candidates re-evaluate against the updated portfolio. Reallocation starts from an explicit historical two-member library (`[0.25, 0.25]`, not counted as new admissions), keeping independent-account total capital `<= 1.0`. Every condition also verifies `ContextConflict` rejection on stale reports followed by fresh re-evaluation and commit.
- **Timing boundaries:** research timing covers initial evaluation, ordered commit, pre-reweight evaluation, weight trial/commit, stale-report rejection, and post-reweight evaluation/commit; lifecycle timing adds setup, load/JIT, registration, export, release, and close. Fixture: 8 synthetic crypto spot assets, 576 base events, 192 decisions, 48 valuation periods, `Admission(weight_steps=4)`.
- **Interpretation:** the first 4-candidate baseline includes initial cold-start/compilation overhead, whereas the 16-candidate rows reflect warmed steady-state workflow cost. Under shared accounts, `[2.0]` scales emitted order quantities rather than implying 200% capital leverage. Targeted smoke passed 32 checks; complete metrics and source SHA-256 digests live in [summary.json](summary.json).
