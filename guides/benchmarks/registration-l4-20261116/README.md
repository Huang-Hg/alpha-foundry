# L4: faster durable array registration

[English](README.md) · [简体中文](README.zh-CN.md)

**On the same L4, factor registration takes half the time and the mining loop is 1.51× faster. Metrics, rewards and admission decisions match exactly for all 512 candidates.** Durable storage and content identity are preserved.

| Workload | Stage | Previous | Current | Speedup |
| --- | --- | ---: | ---: | ---: |
| mlp | Registration | 0.8371 s | 0.4258 s | 1.97× |
| mlp | Mining loop | 3.5131 s | 2.9875 s | 1.18× |
| mlp | Open through close | 5.8719 s | 5.2923 s | 1.11× |
| factor | Registration | 0.8029 s | 0.4015 s | 2.00× |
| factor | Mining loop | 1.1992 s | 0.7929 s | 1.51× |
| factor | Open through close | 1.5033 s | 1.0823 s | 1.39× |

## Changes

- One owned contiguous GPU copy per bounded batch feeds resident evaluation and the durable host mirror. Evaluation does not upload freshly persisted values again.
- Direct pinned host staging expires per chunk. There is no global staging cache or allocation proportional to total candidate count.
- Native SHA-256 consumes buffer views. NPZ streams directly to atomic staging files, removing payload-sized BytesIO and byte copies. Up to four Python IO schedulers handle independent artifacts in input order.
- The GPU cache evicts whole allocations. Mixed hits/misses are assembled together before publication, so old row views cannot pin unbounded batch history.

C++ owns performance-sensitive CPU kernels, CUDA owns batched parallel computation, and Python owns interfaces, scheduling and lifecycle. Hashing uses native hashlib; this patch introduces no Python numerical loop.

## Correctness and space

- After smoke, 79 checks and 6946 subtests pass, covering four markets, PIT, RPC, funding/liquidation, ownership, eviction, reload and failed-write cleanup. The existing missing-price divide warning is preserved in raw logs.
- One old/new run on the same L4: 269 days × 500 equities, 4×64 candidates per model. Complete reports, candidate SHA-256 identities, rewards, learner updates, weights/admission, final portfolios and factor families match exactly; maximum absolute error is zero.
- MLP/factors have 47/74 hard G2 passes, but zero mining selections after correlation filtering. Separate successful public submissions and stale-report CAS checks pass for both. Loop timing includes admission checks, not successful-selection reweight work.
- Current output batch peaks at 32.84 MiB; candidate capacity is 64 and measurement cache is 32 MiB. Sampled whole-GPU peaks change from 924/1214 to 962/1254 MiB, a 38/40 MiB transient ownership/replacement increase. Space remains O(candidate_batch × T × S), independent of total proposals.
- Durable files remain 256/264, approximately 146/151 decimal MB. Cross-session recovery still requires one host mirror and file persistence; this is not end-to-end zero-copy.
- Factor registration falls from about 67.0% to 50.6% of loop time. The remaining stage includes transfer, hashing and file handling; no profiler attribution was run, so it cannot all be labeled disk time.
- Arithmetic, models, probes and thresholds are unchanged. Short-window delayed twins remain empty; this is not robustness or out-of-sample return evidence.
- L4 released; production baselines unchanged. Numerical execution occurred only on Colab.

## Reproduction

The unified entry point remains:

```bash
python -m foundry --parameters '{"search":{"epochs":4,"batch_size":64},"cuda":{"candidate_batch":64,"measurement_cache_mib":32}}' \
  benchmark var/runs/registration-new --suite nonformula --backends cuda
```

Parity and timing (local artifact) · Current plan (local artifact) · Previous plan (local artifact) · Submission (local artifact) · Checks (local artifact) · Source audit (local artifact) · Source (local archive) · Raw logs (local artifact)

[CuPy pinned asynchronous copy contract](https://docs.cupy.dev/en/v13.4.0/reference/generated/cupy.ndarray.html)
