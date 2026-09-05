# Round-robin tournament ranking

Every bot played every other bot 100 times (colours alternating), 156 games in total, at the arena's fast time control (10 s + 0.1 s per side). Ranked by win rate (wins / games); score counts a draw as half a win.

| Rank | Bot | Win rate | Score | W | D | L | Games | Wins as White | Wins as Black | Losses on time |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `03_handcrafted_evaluation` | **100.0%** | 100.0% | 56 | 0 | 0 | 56 | 28 | 28 | 0 |
| 2 | `02_alpha_beta_pruning` | **30.0%** | 64.5% | 30 | 69 | 1 | 100 | 23 | 7 | 0 |
| 3 | `01_negamax` | **0.6%** | 22.8% | 1 | 69 | 86 | 156 | 1 | 0 | 0 |
| 4 | `04_quiescence_search` | **0.0%** | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | `05_iterative_deepening` | **0.0%** | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6 | `06_move_ordering` | **0.0%** | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7 | `07_transposition_table_zobrist` | **0.0%** | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8 | `08_killer_heuristic` | **0.0%** | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 9 | `09_history_heuristic` | **0.0%** | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 10 | `10_principal_variation_search` | **0.0%** | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 11 | `11_null_move_pruning` | **0.0%** | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 12 | `12_late_move_reductions` | **0.0%** | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 13 | `13_static_exchange_evaluation` | **0.0%** | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 14 | `14_futility_pruning_and_razoring` | **0.0%** | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 15 | `15_selective_extensions` | **0.0%** | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 16 | `negamax` | **0.0%** | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Head-to-head (row bot's W-D-L against column bot)

| | 03 | 02 | 01 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 | 13 | 14 | 15 | negamax |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `03_handcrafted_evaluation` | - | 0-0-0 | 56-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 |
| `02_alpha_beta_pruning` | 0-0-0 | - | 30-69-1 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 |
| `01_negamax` | 0-0-56 | 1-69-30 | - | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 |
| `04_quiescence_search` | 0-0-0 | 0-0-0 | 0-0-0 | - | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 |
| `05_iterative_deepening` | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | - | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 |
| `06_move_ordering` | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | - | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 |
| `07_transposition_table_zobrist` | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | - | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 |
| `08_killer_heuristic` | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | - | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 |
| `09_history_heuristic` | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | - | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 |
| `10_principal_variation_search` | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | - | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 |
| `11_null_move_pruning` | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | - | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 |
| `12_late_move_reductions` | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | - | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 |
| `13_static_exchange_evaluation` | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | - | 0-0-0 | 0-0-0 | 0-0-0 |
| `14_futility_pruning_and_razoring` | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | - | 0-0-0 | 0-0-0 |
| `15_selective_extensions` | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | - | 0-0-0 |
| `negamax` | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | 0-0-0 | - |

Column keys: 03 = `03_handcrafted_evaluation`, 02 = `02_alpha_beta_pruning`, 01 = `01_negamax`, 04 = `04_quiescence_search`, 05 = `05_iterative_deepening`, 06 = `06_move_ordering`, 07 = `07_transposition_table_zobrist`, 08 = `08_killer_heuristic`, 09 = `09_history_heuristic`, 10 = `10_principal_variation_search`, 11 = `11_null_move_pruning`, 12 = `12_late_move_reductions`, 13 = `13_static_exchange_evaluation`, 14 = `14_futility_pruning_and_razoring`, 15 = `15_selective_extensions`, negamax = `negamax`

## Head-to-head win rate (row bot's wins / games played against column bot)

| | 03 | 02 | 01 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 | 13 | 14 | 15 | negamax |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `03_handcrafted_evaluation` | - | - | 100% | - | - | - | - | - | - | - | - | - | - | - | - | - |
| `02_alpha_beta_pruning` | - | - | 30% | - | - | - | - | - | - | - | - | - | - | - | - | - |
| `01_negamax` | 0% | 1% | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| `04_quiescence_search` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| `05_iterative_deepening` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| `06_move_ordering` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| `07_transposition_table_zobrist` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| `08_killer_heuristic` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| `09_history_heuristic` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| `10_principal_variation_search` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| `11_null_move_pruning` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| `12_late_move_reductions` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| `13_static_exchange_evaluation` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| `14_futility_pruning_and_razoring` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| `15_selective_extensions` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| `negamax` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |

## Terminations

- checkmate: 87
- threefold_repetition: 69
