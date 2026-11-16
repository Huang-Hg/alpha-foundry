# Non-formula GPU discovery

One public producer interface runs a causal MLP and a GPU tensor factor program.

| Case | Candidates | Accepted | Mine wall (s) | Open through close (s) |
| --- | ---: | ---: | ---: | ---: |
| mlp | 256 | 0 | 3.1989 | 5.6175 |
| factor | 256 | 0 | 0.8810 | 1.1842 |

Times include output registration, host hashing/storage, GPU evaluation, resident feedback, learning, ordered admission and reweighting. MLP prediction fits use causal prefixes; hyperparameter/factor selection uses this research window, so returns are not independent out-of-sample evidence.

[Detailed stages and checks](summary.json) · [Run plan](plan.json)
