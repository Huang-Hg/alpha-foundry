# L4: resident registration and selective persistence

[English](README.md) · [简体中文](README.zh-CN.md)

**Registration is 5.20× / 6.61× faster and stored payloads roughly halve; complete mining speed is essentially unchanged.** All 512 candidates match exactly on metrics, rewards, learning and admission decisions, with zero maximum numerical error.

| Workload | Stage | Previous | Resident |
| --- | --- | ---: | ---: |
| mlp | Registration | 0.4983 s | 0.0958 s |
| mlp | Complete mining loop | 3.1989 s | 3.2169 s |
| mlp | Open through close | 5.6175 s | 5.6180 s |
| factor | Registration | 0.4194 s | 0.0635 s |
| factor | Complete mining loop | 0.8810 s | 0.8827 s |
| factor | Open through close | 1.1842 s | 1.1703 s |

Both versions ran on the same NVIDIA L4 in this session, once per condition: 269 days × 500 US equities, 4 batches × 64 candidates per program, a 32 MiB measurement cache and resident capacity 64. The previous version is the batched durable registration implementation from `2f691a9`.

## Lifecycle

- `register` validates PIT and owns a bounded GPU copy. Only chunk digests return to the host; payloads are not transferred or written to NPZ.
- `release` ends the lease and releases input ownership. Capacity, foreign references and expired references are explicit errors. Session exit releases its remaining leases.
- `retain` saves selected outputs and returns durable references; the original lease still needs release. Commit retains admitted members, and discovery retains final checkpoint references before releasing its batch.
- No archival of every trial, historical GPU cache or implicit spill is added. Artifacts referenced by earlier published checkpoints remain durable for recovery.

## Where time and space went

Checkpoint retention takes 0.6156 s / 0.4134 s, including selection, staging transfer, host hashing and file IO, and remains inside the complete loop. This absorbs the registration savings; a registration speedup is not an end-to-end speedup. Asynchronous stages overlap and are not additive. Open-through-close includes MLP cold model preparation and factor-family judging. Modes always run MLP then factor, so their startup costs are not directly comparable.

| Workload | Artifacts: old → new | Stored MB: old → new | Sampled peak MiB: old → new |
| --- | ---: | ---: | ---: |
| mlp | 256 → 128 | 146.06 → 73.03 | 962 → 928 |
| factor | 264 → 128 | 150.62 → 73.03 | 1254 → 1146 |

Maximum output batch: 32.84 MiB; active candidate rows at completion: zero. Input ownership is `O(candidate_batch × T × S)`, independent of total proposal history; retention staging uses the same bound. Each 4096-byte GPU leaf returns a 32-byte digest. Full payload transfer occurs only for required persistence.

## Correctness and limits

- Smoke passed before 150 test cases and 7206 subchecks were completed. Only the last legacy perpetual reopen case was rerun after explicit retention migration; initial failures and repair logs are preserved.
- Checks cover GPU/hashlib padding and chunk boundaries, future PIT cutoffs, owned copies, capacity, expired leases, RPC, reopen, four markets, funding/liquidation and atomicity after publication failure.
- All numeric fields for 512 candidates and a nine-point factor family match exactly. Only intentionally versioned content hashes, resident scopes and portfolio/context version IDs are excluded.
- MLP/factor G2 passes remain 47/74; correlation screening rejects every final mining selection. A separate resident commit contract covers successful admission. These loop times do not model heavy successful-admission reweight work.
- The short window produces no valid delayed twins; no robustness or out-of-sample return claim follows. Original data publication/revision limitations remain disclosed in DataContract.
- Formats become `research-v14` / `candidate-array-v3` / `candidate-discovery-v3`; old artifacts are explicitly rejected without compatibility shims. Private production baselines are untouched.
- The initial NVRTC header dependency was fixed. Production source stayed unchanged after smoke; subsequent changes corrected tests for the new contract. L4 was released; no numerical tests ran locally.

## Reproduce

```bash
python -m foundry --parameters '{"market":"us_equity","search":{"epochs":4,"batch_size":64},"cuda":{"candidate_batch":64,"measurement_cache_mib":32}}' \
  benchmark var/runs/resident-inputs --suite nonformula --backends cuda
```

Parity and timing (local artifact) · Measured plan (local artifact) · Previous plan (local artifact) · Checks (local artifact) · Targeted repair (local artifact) · Source audit (local artifact) · Measured source (local archive) · Raw evidence (local artifact)
