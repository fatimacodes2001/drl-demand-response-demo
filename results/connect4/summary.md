Task `connect4`, held-out test split, mean ± sample std across seeds.

| method           | seeds | win % vs random | win % vs win/block | loss % vs win/block | net only, win % vs win/block | env steps | train wall (s) |
|------------------|-------|-----------------|--------------------|---------------------|------------------------------|-----------|----------------|
| random           | -     | 48.5            | 3.5                | 96.5                | 3.5                          | -         | -              |
| winblock         | -     | 95.0            | 47.0               | 50.0                | 47.0                         | -         | -              |
| untrained_search | -     | 95.0            | 49.5               | 48.0                | 0.0                          | -         | -              |
| c4_alphazero     | 5     | 96.3 ± 0.8      | 68.5 ± 4.3         | 26.0 ± 3.7          | 45.8 ± 4.6                   | 200210    | 560            |
