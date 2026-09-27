# Common-area energy models: model card (generated)

Generated 2026-09-26T23:21:53Z by scripts/train_energy_models.py. Every number is copied from eval/energy/*.json.

| model | data | split | n test | positives | metric | model | best baseline | baseline | model wins | 95% CI of difference (days) |
|---|---|---|---|---|---|---|---|---|---|---|
| M1 presence now | ROBOD room 1 (Lecture room) | day-blocked 70/30 | 2592 | 55 | F1 (occupied) | 0.072 | schedule_168 | 0.065 | yes | -0.012 to +0.075 (no clear difference) |
| M1 presence now | ROBOD room 2 (Lecture room) | day-blocked 70/30 | 2592 | 555 | F1 (occupied) | 0.580 | schedule_168 | 0.491 | yes | +0.018 to +0.156 (model better) |
| M1 presence now | ROBOD room 3 (Office space) | day-blocked 70/30 | 2592 | 789 | F1 (occupied) | 0.803 | schedule_48 | 0.793 | yes | -0.013 to +0.028 (no clear difference) |
| M1 presence now | ROBOD room 4 (Office space) | day-blocked 70/30 | 4320 | 2522 | F1 (occupied) | 0.911 | schedule_168 | 0.910 | yes | -0.011 to +0.013 (no clear difference) |
| M1 presence now | ROBOD room 5 (Library space) | day-blocked 70/30 | 4320 | 1535 | F1 (occupied) | 0.818 | schedule_48 | 0.819 | no | -0.043 to +0.033 (no clear difference) |
| M2 presence +1 h (deployable) | ROBOD room 1 | day-blocked 70/30 | 2484 | 55 | F1 (occupied) | 0.086 | schedule_168 | 0.065 | yes | -0.001 to +0.084 (no clear difference) |
| M2 presence +1 h (oracle) | ROBOD room 1 | day-blocked 70/30 | 2484 | 55 | F1 (occupied) | 0.138 | persistence_oracle | 0.582 | no | -0.567 to +0.000 (no clear difference) |
| M2 presence +1 h (deployable) | ROBOD room 2 | day-blocked 70/30 | 2484 | 555 | F1 (occupied) | 0.517 | schedule_168 | 0.491 | yes | -0.054 to +0.106 (no clear difference) |
| M2 presence +1 h (oracle) | ROBOD room 2 | day-blocked 70/30 | 2484 | 555 | F1 (occupied) | 0.740 | persistence_oracle | 0.814 | no | -0.149 to -0.033 (baseline better) |
| M2 presence +1 h (deployable) | ROBOD room 3 | day-blocked 70/30 | 2484 | 789 | F1 (occupied) | 0.793 | schedule_48 | 0.793 | no | -0.019 to +0.021 (no clear difference) |
| M2 presence +1 h (oracle) | ROBOD room 3 | day-blocked 70/30 | 2484 | 789 | F1 (occupied) | 0.815 | persistence_oracle | 0.853 | no | -0.167 to +0.034 (no clear difference) |
| M2 presence +1 h (deployable) | ROBOD room 4 | day-blocked 70/30 | 4140 | 2486 | F1 (occupied) | 0.918 | schedule_48 | 0.923 | no | -0.015 to +0.003 (no clear difference) |
| M2 presence +1 h (oracle) | ROBOD room 4 | day-blocked 70/30 | 4140 | 2486 | F1 (occupied) | 0.931 | persistence_oracle | 0.941 | no | -0.086 to +0.030 (no clear difference) |
| M2 presence +1 h (deployable) | ROBOD room 5 | day-blocked 70/30 | 4140 | 1535 | F1 (occupied) | 0.661 | schedule_48 | 0.819 | no | -0.281 to -0.063 (baseline better) |
| M2 presence +1 h (oracle) | ROBOD room 5 | day-blocked 70/30 | 4140 | 1535 | F1 (occupied) | 0.889 | persistence_oracle | 0.884 | yes | -0.066 to +0.045 (no clear difference) |
| M3 garage day-ahead (no weather) | BDG2 20 parking meters | train 2016 / test 2017 | 174258 | - | median CV(RMSE) (lower is better) | 0.091 | lag24 | 0.096 | yes | - |

UCI 357 test2 (forward in time) F1: majority 0.000, schedule_48 0.845, logreg:with_light (leaky) 0.983, hgb:with_light (leaky) 0.957, logreg:no_light 0.669, hgb:no_light 0.772

Labels: REAL = public measured data; SEMI-SYNTHETIC / INJECTED / SIMULATED are marked in each JSON.
