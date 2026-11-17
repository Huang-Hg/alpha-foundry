# L4: removing duplicate retention work

[English](README.md) · [简体中文](README.zh-CN.md)

**With identical saved payloads, the complete MLP mining loop is 1.14× faster and the non-formula factor loop is 1.38× faster.** Open-through-close improves by 1.08× / 1.26×; all 512 candidates match exactly on metrics, rewards, learning and admission decisions.

| Program | Stage | Previous | Current | Speedup |
| --- | --- | ---: | ---: | ---: |
| mlp | Retention and persistence | 0.5560 s | 0.2038 s | 2.73× |
| mlp | Complete mining loop | 3.2526 s | 2.8479 s | 1.14× |
| mlp | Open through close | 5.5941 s | 5.1692 s | 1.08× |
| factor | Retention and persistence | 0.4122 s | 0.1773 s | 2.33× |
| factor | Complete mining loop | 0.8761 s | 0.6368 s | 1.38× |
| factor | Open through close | 1.1505 s | 0.9119 s | 1.26× |

Reference: `0f8d249`. Both versions ran once on the same NVIDIA L4 in this session: 269 days × 500 US equities, 4×64 candidates per program. Complete timing includes registration, generation, evaluation, learning, ordered admission, required retention, checkpoint publication and resource closure. Modes run in a fixed MLP-then-factor order; startup costs across programs are not directly comparable. Stages overlap and are not additive.

## Work removed

- New writes reuse registration-computed content identities, removing repeated CPU hashing of every 4 KiB leaf. Reading existing artifacts still validates content and provenance.
- Selections group by original lease allocation; contiguous candidates borrow GPU slices and only noncontiguous selections gather.
- Host PIT cutoffs stay on the host, removing their upload, concatenation and download round trip.
- `ArrayStore.put(batch, sources)` explicitly receives the validated batch and aligned identities. Callers and validation migrate together without the old signature. High-level Session methods, artifact formats and content digests stay unchanged.

## Durability and space

Checkpoints still publish after each batch, with identical archive capacity and retained candidates. Both modes save 128 artifacts: 73,032,192 / 73,029,632 bytes, exactly matching the reference. Required persistence completes before return; no write-behind backlog is introduced. Batch capacity remains 64 and the measurement cache remains 32 MiB, without an added history cache.

Sampled GPU peaks: MLP 928 → 928 MiB; factor 1220 → 1146 MiB. Active candidate rows at completion: zero. Space remains bounded by batch size and existing cache budgets, independently of cumulative proposal count.

## Validation and limits

After smoke, all 131 tests and 7108 subchecks passed without a rerun. Coverage includes no rehash on initial persistence, contiguous slice reuse, PIT, four markets, RPC, funding/liquidation, corrupted artifact rejection, expiration, recovery and atomic commit. Complete trajectory comparison has zero maximum numerical error; content digests and context/portfolio versions also match. Only random runtime scopes are excluded.

G2 passes remain 47 for MLP and 74 for factors; correlation screening leaves zero final mining selections. Separate commit tests cover successful admission, so this timing does not represent heavy successful-selection reweighting. The short window has no valid delayed twins and supplies no robustness or out-of-sample return evidence. One run per condition provides no performance confidence interval. The existing missing-price divide warning remains in raw logs.

Numerical acceptance ran only on Colab. L4 was released; private production baselines were not rewritten. No push or merge was performed.

## Reproduce

```bash
python -m foundry --parameters '{"market":"us_equity","search":{"epochs":4,"batch_size":64},"cuda":{"candidate_batch":64,"measurement_cache_mib":32}}' \
  benchmark var/runs/retention --suite nonformula --backends cuda
```

Timing and exact parity (local artifact) · Measured plan (local artifact) · Previous plan (local artifact) · Contracts (local artifact) · Source audit (local artifact) · Measured source (local archive) · Full trajectories and logs (local artifact)
