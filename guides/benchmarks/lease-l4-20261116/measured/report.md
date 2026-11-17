# Non-formula GPU discovery

One public producer interface runs a causal MLP and a GPU tensor factor program.

| Case | Candidates | Accepted | Mine wall (s) | Open through close (s) |
| --- | ---: | ---: | ---: | ---: |
| mlp | 256 | 0 | 3.2169 | 5.6180 |
| factor | 256 | 0 | 0.8827 | 1.1703 |

Times include resident registration, GPU hashing/evaluation, resident feedback, learning, ordered admission, reweighting and selective persistence. Registration transfers only digests; checkpoint and admitted artifacts are included in the full lifecycle. MLP prediction fits use causal prefixes; hyperparameter/factor selection uses this research window, so returns are not independent out-of-sample evidence.

[Detailed stages and checks](summary.json) · Run plan (local artifact)
