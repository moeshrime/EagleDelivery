# Eagle Delivery

COSC 461/561 Introduction to Artificial Intelligence, Project 1 (Eastern Michigan University, Fall 2026).

Eagle Delivery is a simulated delivery robot that carries a package from one EMU building entrance to another on a grid map. It works in three stages:

1. **Search (Part A):** plan the route on a fully known map with **A\*** (Manhattan heuristic) and compare it with the provided **BFS** baseline.
2. **Reinforcement learning (Part B):** learn a delivery policy from experience alone with **tabular Q-learning** and **epsilon-greedy** exploration.
3. **Experiment (Part C):** a custom campus map (`maps/my_campus.txt`) and a controlled experiment on the exploration rate epsilon.

The full write-up is in [`report/EagleDelivery_Report.pdf`](report/EagleDelivery_Report.pdf).

## Requirements

- Python 3.9 or newer. The simulator, search and learning code use only the standard library.
- `matplotlib` is needed only to redraw the plots with `analyze_results.py` (`pip install matplotlib`).

## Repository layout

| Path | Contents |
|---|---|
| `agents.py` | BFS (provided) plus my four completed TODOs: `manhattan`, `astar`, `epsilon_greedy`, `q_learning_update` |
| `campus.py` | Map loading, movement, rewards, resets (provided) |
| `run_project.py` | Show maps, run search, train and evaluate Q-learning, save CSVs (provided) |
| `summarize_results.py` | Per-folder evaluation summary and `learning_curve.csv` (provided) |
| `search_table.py` | Runs BFS and A* on a set of tasks and saves one comparison CSV |
| `analyze_results.py` | Builds `results/training_stats.csv` and the report figures in `results/plots/` |
| `maps/` | `emu_campus.txt`, `emu_detour.txt`, and my map `my_campus.txt` |
| `results/` | All raw training/evaluation CSVs, summaries, search outputs, and plots |
| `report/` | The PDF report and its HTML source |
| `tests/`, `examples/` | Public tests and the API demo (provided) |

## Run commands

Run every command from the repository folder.

**Check the code**

```bash
python3 -m unittest discover -v          # all 14 tests pass
python3 run_project.py show
python3 run_project.py show --map maps/my_campus.txt
```

**Part A: search**

```bash
python3 run_project.py search --algorithm bfs   --start S --goal W
python3 run_project.py search --algorithm astar --start S --goal W
python3 run_project.py search --algorithm astar --map maps/emu_detour.txt --start L --goal D
python3 search_table.py                  # six required tasks -> results/search/search_results.csv
python3 search_table.py --map maps/my_campus.txt --tasks SW --output results/search/custom_search.csv
```

**Part B: baseline Q-learning (starter defaults)**

```bash
python3 run_project.py train --map maps/emu_campus.txt --start S --goal W --output results/baseline_SW
python3 run_project.py train --map maps/emu_campus.txt --start L --goal D --output results/baseline_LD
python3 run_project.py train --map maps/emu_campus.txt --start H --goal P --output results/baseline_HP
python3 summarize_results.py results/baseline_SW
python3 summarize_results.py results/baseline_LD
python3 summarize_results.py results/baseline_HP
```

**Part C: epsilon experiment on my map**

```bash
python3 run_project.py search --map maps/my_campus.txt --algorithm astar
python3 run_project.py train --map maps/my_campus.txt --epsilon 0.05 --output results/custom_e005
python3 run_project.py train --map maps/my_campus.txt --epsilon 0.2  --output results/custom_e020
python3 run_project.py train --map maps/my_campus.txt --epsilon 0.5  --output results/custom_e050
python3 summarize_results.py results/custom_e005
python3 summarize_results.py results/custom_e020
python3 summarize_results.py results/custom_e050
```

**Tables and plots**

```bash
python3 analyze_results.py               # results/training_stats.csv + results/plots/*.png
```

(`--map maps/emu_campus.txt` is the default map; passing it explicitly just records a relative path in `summary.csv`.)

`run_project.py` refuses to overwrite a folder that already has CSVs. To rerun an experiment, use a new `--output` folder.

## Experiment settings

| Setting | Part B baseline | Part C experiment |
|---|---|---|
| Map | `emu_campus.txt` | `my_campus.txt` |
| Tasks | S→W, L→D, H→P | S→W |
| Training episodes | 2,000 per seed | 2,000 per seed |
| Seeds | 0, 1, 2 (fresh Q-table each) | 0, 1, 2 for each setting |
| Evaluation | 100 greedy episodes (epsilon 0, frozen Q) | same |
| Max actions per episode | 200 | 200 |
| Learning rate alpha | 0.3 | 0.3 |
| Discount gamma | 0.95 | 0.95 |
| Exploration epsilon | 0.2 | **0.05, 0.2, 0.5** (the only factor changed) |
| Slip probability | 0.0 | 0.0 |

## Results at a glance

**Part A: BFS vs A\*** (`results/search/search_results.csv`)

| Map | Task | BFS length | A* length | BFS expanded | A* expanded |
|---|---|---|---|---|---|
| Baseline | S→W | 18 | 18 | 67 | 22 |
| Baseline | L→D | 22 | 22 | 79 | 62 |
| Baseline | H→P | 16 | 16 | 41 | 16 |
| Detour | S→W | 26 | 26 | 70 | 51 |
| Detour | L→D | 22 | 22 | 65 | 60 |
| Detour | H→P | 16 | 16 | 41 | 16 |
| My campus | S→W | 26 | 26 | 61 | 34 |

**Part B:** all 9 baseline runs reach 100% evaluation success along the BFS-optimal route (18, 22 and 16 actions).

**Part C:** epsilon made no difference to the final greedy policy (100% success, 26 actions, for all 9 runs). It strongly affected training cost: mean training actions over the last 500 episodes were 27.4 / 32.8 / 53.9 for epsilon 0.05 / 0.2 / 0.5. The smallest epsilon also learned fastest.

## my_campus.txt

```
#####################
#S........#########W#
#####.#############.#
#####....L#########.#
#####.#############.#
#####...............#
#####.#########.###.#
#####.#########.###D#
#####.#########.#####
#...............#####
#.#############.#####
#H#############P#####
#####################
```

- **Loop:** rows 5 and 9 joined by columns 5 and 15. The top of the loop is the 26-move route, and the bottom is 8 moves longer.
- **Dead ends:** the top corridor that points at W but is walled off, plus the L, D, H and P spurs.
