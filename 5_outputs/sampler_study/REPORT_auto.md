# Sampler study: automatic report (2026-10-12 22:26)

Criterion: chains (or particles) with log-likelihood above -14395; mode A at -14385 (log posterior -21124, typical chain -21108).

## Round 2 (this study folder)

| run | status | above -14395 | median loglik | best loglik | best logpost | families above | options |
|---|---|---|---|---|---|---|---|
| smc3 | done | 26/256 | -14491 | -14302 | nan | A:0,B:0,C:0,D:0,E:0 |  |
| crit_long | done | 0/10 | -14436 | -14396 | -21153 |  | n=10 t0=0.001 steps=16 ladder=0.001,0.003,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.12,0.15,0.2,0.3,0.4,0.6,0.8,1.0 tune=150 draws=20 crit=0.03,0.4 cri |
| smc3r | done | 36/256 | -14449 | -14272 | nan | A:0,B:0,C:0,D:0,E:0 |  |
| smc3_top | done | | | | | | RESULT smc3_top: exported the 10 best particles of smc3r (rejuv.npz), loglik [-14272.0, -14276.0, -14301.0, -14306.0, -14310.0, -14317.0, -14321.0, -14335.0, -1 |
| pol_smc3 | done | 0/10 | -14452 | -14424 | -21180 |  | n=10 t0=0.001 steps=16 tune=150 draws=20 crit_tune=300 crit_draws=50 init=jitter final_tune=1000 final_draws=300 polish_from=smc3_top:0,1,2,3,4,5,6,7,8,9 seed=1 |
| pol_crit_long | done | | | | | | SKIPPED pol_crit_long: crit_long has no chain above -14395 ((3, -14396.1)) |
| crit_kb2 | done | 2/10 | -14432 | -14352 | -21083 | D,E | n=10 t0=0.001 steps=16 ladder=0.001,0.003,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.12,0.15,0.2,0.3,0.4,0.6,0.8,1.0 tune=150 draws=20 crit=0.03,0.4 cri |
| crit_resample | done | 0/10 | -14428 | -14399 | -21146 |  | n=10 t0=0.001 steps=16 ladder=0.001,0.003,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.12,0.15,0.2,0.3,0.4,0.6,0.8,1.0 tune=150 draws=20 crit=0.03,0.4 cri |
| pol_crit_kb2 | done | 6/10 | -14392 | -14381 | -21128 | A,C,D | n=10 t0=0.001 steps=16 tune=150 draws=20 crit_tune=300 crit_draws=50 init=jitter final_tune=1000 final_draws=300 polish_from=crit_kb2:4 seed=72100 |
| crit_wide | done | 1/10 | -14416 | -14395 | -21143 | E | n=10 t0=0.001 steps=16 ladder=0.001,0.003,0.01,0.015,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.12,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.8,1.0 tune=150 draws=20 cr |
| pop_b1 | done | | | | | | RESULT pop_b1: stopped at T=0.4; stage means [-17084.0, -17086.0, -17089.0, -17092.0, -17097.0, -17111.0, -17117.0, -17120.0, -17144.0, -17157.0]; 2045 s |
| pop_b2 | done | | | | | | RESULT pop_b2: stopped at T=0.4; stage means [-17045.0, -17095.0, -17098.0, -17100.0, -17103.0, -17108.0, -17132.0, -17133.0, -17136.0, -17160.0]; 2117 s |
| pop_b3 | done | | | | | | RESULT pop_b3: stopped at T=0.4; stage means [-17071.0, -17078.0, -17089.0, -17099.0, -17108.0, -17109.0, -17109.0, -17109.0, -17111.0, -17176.0]; 2100 s |
| pop_finish | done | 0/10 | -14434 | -14397 | -21138 |  | n=10 t0=0.001 steps=16 ladder=0.001,0.003,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.12,0.15,0.2,0.3,0.4,0.6,0.8,1.0 tune=150 draws=20 crit=0.03,0.4 cri |
| pol_pop | done | | | | | | SKIPPED pol_pop: pop_finish has no chain above -14395 ((6, -14396.9)) |
| crit_s4 | done | 0/10 | -14436 | -14401 | -21143 |  | n=10 t0=0.001 steps=16 ladder=0.001,0.003,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.12,0.15,0.2,0.3,0.4,0.6,0.8,1.0 tune=150 draws=20 crit=0.03,0.4 cri |
| control_nuts2 | done | 5/10 | -14411 | -14373 | -21112 | B,D,E | n=10 t0=0.001 steps=16 ladder=1.0 tune=150 draws=20 crit_tune=300 crit_draws=50 init=jitter final_tune=1800 final_draws=300 seed=191000 |
| crit_s5 | done | 0/10 | -14454 | -14402 | -21153 |  | n=10 t0=0.001 steps=16 ladder=0.001,0.003,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.12,0.15,0.2,0.3,0.4,0.6,0.8,1.0 tune=150 draws=20 crit=0.03,0.4 cri |
| control_nutpie1 | done | 4/10 | -14413 | -14374 | -21123 | B,D,E | n=10 t0=0.001 steps=16 ladder=1.0 tune=150 draws=20 crit_tune=300 crit_draws=50 init=jitter final_tune=1800 final_draws=300 seed=201000 sampler=nutpie |
| control_nuts3 | done | 2/10 | -14423 | -14389 | -21140 | D,E | n=10 t0=0.001 steps=16 ladder=1.0 tune=150 draws=20 crit_tune=300 crit_draws=50 init=jitter final_tune=1800 final_draws=300 seed=192000 sampler=pymc |
| crit_narrow | done | 4/10 | -14412 | -14366 | -21123 | B,D,E | n=10 t0=0.001 steps=16 ladder=0.001,0.003,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.14,0.16,0.18,0.2,0.3,0.4,0.6,0.8,1.0 tune=150 draws=20 cr |
| crit_s8 | done | 0/10 | -14443 | -14401 | -21150 |  | n=10 t0=0.001 steps=16 ladder=0.001,0.003,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.12,0.15,0.2,0.3,0.4,0.6,0.8,1.0 tune=150 draws=20 crit=0.03,0.4 cri |
| control_nuts4 | done | 7/10 | -14385 | -14364 | -21105 | B,D,E | n=10 t0=0.001 steps=16 ladder=1.0 tune=150 draws=20 crit_tune=300 crit_draws=50 init=jitter final_tune=1800 final_draws=300 seed=193000 sampler=pymc |
| crit_s9 | done | 2/10 | -14428 | -14363 | -21118 | B,E | n=10 t0=0.001 steps=16 ladder=0.001,0.003,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.12,0.15,0.2,0.3,0.4,0.6,0.8,1.0 tune=150 draws=20 crit=0.03,0.4 cri |
| control_long | done | 5/10 | -14405 | -14354 | -21109 | B,D | n=10 t0=0.001 steps=16 ladder=1.0 tune=150 draws=20 crit_tune=300 crit_draws=50 init=jitter final_tune=4000 final_draws=300 seed=203000 sampler=pymc |
| control_nutpie2 | done | 4/10 | -14414 | -14370 | -21123 | A,B,D | n=10 t0=0.001 steps=16 ladder=1.0 tune=150 draws=20 crit_tune=300 crit_draws=50 init=jitter final_tune=1800 final_draws=300 seed=202000 sampler=nutpie |
| control_nuts5 | done | 3/10 | -14414 | -14379 | -21125 | B,C | n=10 t0=0.001 steps=16 ladder=1.0 tune=150 draws=20 crit_tune=300 crit_draws=50 init=jitter final_tune=1800 final_draws=300 seed=194000 sampler=pymc |
| control_nutpie3 | done | 4/10 | -14406 | -14370 | -21119 | A,C,D | n=10 t0=0.001 steps=16 ladder=1.0 tune=150 draws=20 crit_tune=300 crit_draws=50 init=jitter final_tune=4000 final_draws=300 seed=204000 sampler=nutpie |
| crit_narrow2 | done | 2/10 | -14414 | -14391 | -21133 | B,C | n=10 t0=0.001 steps=16 ladder=0.001,0.003,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.14,0.16,0.18,0.2,0.3,0.4,0.6,0.8,1.0 tune=150 draws=20 cr |
| control_nutpie4 | done | 3/10 | -14404 | -14374 | -21111 | B,D | n=10 t0=0.001 steps=16 ladder=1.0 tune=150 draws=20 crit_tune=300 crit_draws=50 init=jitter final_tune=1800 final_draws=300 seed=205000 sampler=nutpie |
| crit_narrow3 | done | 1/10 | -14409 | -14392 | -21139 | C | n=10 t0=0.001 steps=16 ladder=0.001,0.003,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.14,0.16,0.18,0.2,0.3,0.4,0.6,0.8,1.0 tune=150 draws=20 cr |
| control_nuts6 | done | 3/10 | -14433 | -14363 | -21111 | B,C,D | n=10 t0=0.001 steps=16 ladder=1.0 tune=150 draws=20 crit_tune=300 crit_draws=50 init=jitter final_tune=1800 final_draws=300 seed=195000 sampler=pymc |

## Round 1 (2026-10-08/09)

| run | recipe | above -14395 | median | best loglik | best logpost | note |
|---|---|---|---|---|---|---|
| hot_slow3b | geometric 1e-3, 16 stages | 0/10 | -14446 | -14406 | -21159 | fixed protocol (no jitter, long final) |
| slow_long | 30 stages, tune 300 | 0/10 | -14446 | -14403 | -21153 | |
| hotter | 1e-4, 20 stages | 2/10 | -14434 | -14392 | -21145 | |
| control_nuts | plain NUTS, same budget | 1/10 | -14437 | -14386 | -21140 | the baseline |
| crit_select | crit window 0.03-0.4, keep-best 3 on stage means | 3/10 | -14422 | -14357 | -21097 | polished: 6/10 stay above, E-like family |
| crit_s2 | same, seed 2 | 2/10 | -14398 | -14374 | -21121 | |
| crit_s3 | same, seed 3 | 2/10 | -14424 | -14381 | -21122 | |
| crit_kb5 | keep-best 5 | 1/10 | -14425 | -14377 | -21126 | stronger selection loses diversity |
| pt48 | parallel tempering, 48 replicas 0.1-1 | n/a | cold -14382 | | | no round trip in 4500 iterations; swaps 7% near T=1 |
| smc 10x10 / 50x20 | tempered SMC | n/a | | | | too little mutation / fell into phi<0 (bug fixed) |

## Runner log (last 30 lines)

```
2026-10-11 09:15:10 END   control_nuts4 rc=0 3727s RESULT control_nuts4: 7/10 chains above -14395, 6 at A's level or better; median loglik -14379; best loglik -14364, best log posterior -21105 (A -14385 / -21124); 3723 s
2026-10-11 09:15:10 START anneal crit_s9 --ladder 0.001,0.003,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.12,0.15,0.2,0.3,0.4,0.6,0.8,1.0 --crit 0.03,0.4 --crit-tune 300 --crit-draws 50 --select-mean --keep-best 3 --seed 182000
2026-10-11 10:47:47 END   crit_s9 rc=0 5557s RESULT crit_s9: 2/10 chains above -14395, 1 at A's level or better; median loglik -14424; best loglik -14363, best log posterior -21118 (A -14385 / -21124); 5553 s
2026-10-11 10:47:47 START anneal control_long --ladder 1.0 --final-tune 4000 --final-draws 300 --seed 203000
2026-10-11 12:55:13 END   control_long rc=0 7646s RESULT control_long: 5/10 chains above -14395, 2 at A's level or better; median loglik -14400; best loglik -14354, best log posterior -21109 (A -14385 / -21124); 7642 s
2026-10-11 12:55:13 START anneal control_nutpie2 --ladder 1.0 --sampler nutpie --final-tune 1800 --final-draws 300 --seed 202000
2026-10-11 14:06:00 END   control_nutpie2 rc=0 4246s RESULT control_nutpie2: 4/10 chains above -14395, 2 at A's level or better; median loglik -14412; best loglik -14370, best log posterior -21123 (A -14385 / -21124); 4242 s
2026-10-11 14:06:00 START anneal control_nuts5 --ladder 1.0 --final-tune 1800 --final-draws 300 --seed 194000
2026-10-11 15:07:03 END   control_nuts5 rc=0 3663s RESULT control_nuts5: 3/10 chains above -14395, 1 at A's level or better; median loglik -14410; best loglik -14379, best log posterior -21125 (A -14385 / -21124); 3659 s
2026-10-11 15:07:03 START anneal control_nutpie3 --ladder 1.0 --sampler nutpie --final-tune 4000 --final-draws 300 --seed 204000
2026-10-11 17:33:05 END   control_nutpie3 rc=0 8762s RESULT control_nutpie3: 4/10 chains above -14395, 2 at A's level or better; median loglik -14405; best loglik -14370, best log posterior -21119 (A -14385 / -21124); 8757 s
2026-10-11 17:33:05 START anneal crit_narrow2 --ladder 0.001,0.003,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.14,0.16,0.18,0.2,0.3,0.4,0.6,0.8,1.0 --crit 0.04,0.2 --crit-tune 300 --crit-draws 50 --select-mean --keep-best 3 --seed 163000
2026-10-11 19:09:44 END   crit_narrow2 rc=0 5799s RESULT crit_narrow2: 2/10 chains above -14395, 0 at A's level or better; median loglik -14414; best loglik -14391, best log posterior -21133 (A -14385 / -21124); 5794 s
2026-10-11 19:09:44 START anneal control_nutpie4 --ladder 1.0 --sampler nutpie --final-tune 1800 --final-draws 300 --seed 205000
2026-10-11 20:19:56 END   control_nutpie4 rc=0 4212s RESULT control_nutpie4: 3/10 chains above -14395, 3 at A's level or better; median loglik -14402; best loglik -14374, best log posterior -21111 (A -14385 / -21124); 4207 s
2026-10-11 20:19:56 START anneal crit_narrow3 --ladder 0.001,0.003,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.14,0.16,0.18,0.2,0.3,0.4,0.6,0.8,1.0 --crit 0.04,0.2 --crit-tune 300 --crit-draws 50 --select-mean --keep-best 3 --seed 164000
2026-10-11 21:55:55 END   crit_narrow3 rc=0 5759s RESULT crit_narrow3: 1/10 chains above -14395, 0 at A's level or better; median loglik -14409; best loglik -14392, best log posterior -21139 (A -14385 / -21124); 5755 s
2026-10-11 21:55:55 START anneal control_nuts6 --ladder 1.0 --final-tune 1800 --final-draws 300 --seed 195000
2026-10-11 22:58:11 END   control_nuts6 rc=0 3736s RESULT control_nuts6: 3/10 chains above -14395, 1 at A's level or better; median loglik -14421; best loglik -14363, best log posterior -21111 (A -14385 / -21124); 3732 s
2026-10-12 00:58:26 no job for 2 h: exiting
2026-10-12 00:58:26 runner exits
2026-10-12 10:01:42 runner started (pid 34035)
2026-10-12 12:01:56 no job for 2 h: exiting
2026-10-12 12:01:56 runner exits
2026-10-12 16:15:47 runner started (pid 35320)
2026-10-12 18:16:01 no job for 2 h: exiting
2026-10-12 18:16:01 runner exits
2026-10-12 18:46:01 runner started (pid 36125)
2026-10-12 20:46:15 no job for 2 h: exiting
2026-10-12 20:46:15 runner exits
```
