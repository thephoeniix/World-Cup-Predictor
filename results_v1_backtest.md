# V1 Backtest Results — Elo + Host Advantage vs Baselines

Backtest across the 2010, 2014, 2018, 2022, and 2026 (group stage, partial) World Cups.
Train = every match strictly before each tournament's cutoff date (no leakage). Metric = RPS (lower is better), validated with paired bootstrap (10,000 resamples).

Models compared (same Poisson engine, only the formula changes):

| Model | Formula |
|---|---|
| `base_rate` | `goals ~ is_home` |
| `elo_only` (Elo puro) | `goals ~ is_home + elo_diff` |
| `elo_host` (current model) | `goals ~ is_home + elo_diff + host_diff` |

## Mean RPS by World Cup and model

| wc_year | base_rate | elo_host | elo_only |
|---|---|---|---|
| 2010 | 0.251968 | 0.209238 | 0.209238 |
| 2014 | 0.242522 | 0.200889 | 0.203096 |
| 2018 | 0.258357 | 0.235374 | 0.222706 |
| 2022 | 0.238560 | 0.216408 | 0.220853 |
| 2026* | 0.183720 | 0.196506 | 0.185297 |

\* 2026 is partial: only 28 of 72 group-stage fixtures have been played as of this backtest.

## Pooled paired-bootstrap comparison (all 284 scoreable matches)

| Comparison | ΔRPS | 95% CI | p-value | Significant? |
|---|---|---|---|---|
| `elo_host` vs `base_rate` (tasa base) | −0.028 | [−0.046, −0.009] | 0.003 | **Yes** |
| `elo_host` vs `elo_only` (Elo puro) | +0.002 | [−0.001, 0.007] | 0.24 | No |

## Per-World-Cup bootstrap detail

**`elo_host` vs `elo_only`:**

| wc_year | observed_diff | ci95 | p_value | significant | n_matches |
|---|---|---|---|---|---|
| 2010 | ~0.0 | (~0, ~0) | 0.154 | False | 64 |
| 2014 | −0.00221 | (−0.00551, 0.00055) | 0.142 | False | 64 |
| 2018 | +0.01267 | (−0.00154, 0.02952) | 0.064 | False | 64 |
| 2022 | −0.00445 | (−0.01054, 0.00000) | 0.085 | False | 64 |
| 2026 | +0.01121 | (−0.00519, 0.03049) | 0.207 | False | 28 |

**`elo_host` vs `base_rate`:**

| wc_year | observed_diff | ci95 | p_value | significant | n_matches |
|---|---|---|---|---|---|
| 2010 | −0.04273 | (−0.08002, −0.00470) | 0.029 | True | 64 |
| 2014 | −0.04163 | (−0.07040, −0.01243) | 0.004 | True | 64 |
| 2018 | −0.02298 | (−0.06426, 0.01726) | 0.253 | False | 64 |
| 2022 | −0.02215 | (−0.06495, 0.02365) | 0.333 | False | 64 |
| 2026 | +0.01279 | (−0.03944, 0.06701) | 0.654 | False | 28 |

## Takeaways

- **V1's sanity check passes:** `elo_host` beats the no-information baseline (`base_rate`) when pooled across all World Cups (p = 0.003). Per blueprint: "si no [le gana], hay un bug" — confirmed, no bug.
- **`host_diff` isn't pulling weight yet:** `elo_host` is statistically indistinguishable from pure Elo (p = 0.24 pooled, never significant in any single World Cup). Root cause: `host_diff` is only nonzero during a host nation's own World Cup matches — 3 to 18 historical occurrences depending on the cutoff year. For the 2010 backtest specifically it's constant zero in training (no host data exists before the first dict-covered host year), so `elo_host` and `elo_only` produce identical predictions that year.
- **Not a dead end:** this is a proxy-feature sparsity issue specific to the minimal V1 implementation, not evidence that home advantage doesn't matter. Model B's real localía treatment (alongside market value and recent form) arrives in V2 and should have more to work with.

## Repo changes behind this run

- `src/backtest.py`: fixed a std=0 crash in `standardize()` (host_diff is constant zero in 2010 training data), added `FORMULAS` (`base_rate`/`elo_only`/`elo_host`) and `compare_models()`.
- `src/metrics.py`: implemented from scratch — RPS, Brier, log loss, ECE, paired bootstrap, `summarize_models()`, `bootstrap_compare_models()`.
- `src/poisson_model.py`: added `match_outcome_probs()` (lambdas → P(home/draw/away) via independent-Poisson grid summation).
- `.gitignore`: added; untracked `__pycache__`.
