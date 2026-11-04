Task `dr`, held-out test split, mean ± sample std across seeds.

| method          | seeds | test saving $/day | % of oracle saving | env steps | train wall (s) |
|-----------------|-------|-------------------|--------------------|-----------|----------------|
| no_control      | -     | 0.00              | 0.0                | -         | -              |
| random          | -     | -5.72             | -59.5              | -         | -              |
| rule_based      | -     | 0.34              | 3.6                | -         | -              |
| oracle_discrete | -     | 8.78              | 91.2               | -         | -              |
| alphazero       | 5     | 7.91 ± 0.13       | 82.2 ± 1.3         | 30144     | 102            |
| ddpg            | 5     | 9.08 ± 0.04       | 94.3 ± 0.4         | 60000     | 93             |
| dqn             | 5     | 8.49 ± 0.08       | 88.2 ± 0.8         | 60000     | 52             |
| per_ddpg        | 5     | 8.87 ± 0.18       | 92.2 ± 1.9         | 60000     | 120            |
| ppo             | 5     | 8.37 ± 0.08       | 87.0 ± 0.8         | 1000704   | 68             |
