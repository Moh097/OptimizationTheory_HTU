# Task 2 – Meta‑heuristic Hyper‑parameter Optimisation  
Optimising **RandomForestClassifier** hyper‑parameters with Genetic
Algorithm (GA), Particle Swarm Optimisation (PSO) and
NSGA‑II (multi‑objective).

---

## 1. Project Overview

| Mode | Algorithm(s) | Objective(s) | Command |
|------|--------------|--------------|---------|
| **Default** | GA → PSO (pick the best) | *Accuracy ↑* | `python main.py` |
| **Multi‑objective** | NSGA‑II | *Accuracy ↑* & *Model size ↓* (`n_estimators`) | `python main.py --multi` |

The workflow:

1. **Data ingestion** – Arabic Q‑and‑A dataset (`data/data_3.csv`) is
   pre‑processed with a TF‑IDF n‑gram vectoriser (1‑ and 2‑grams).

2. **Single objective**  
   *   GA searches 20 generations, population 30.  
   *   PSO refines the search for 20 iterations, swarm 30.  
   *   The better of the two solutions trains the final RF model.

3. **Multi‑objective (NSGA‑II)**  
   *   Population 40, 30 generations.  
   *   Returns the **Pareto front** (non‑dominated set).  
   *   By default, the model with the **highest accuracy** on that front
       is chosen; change the selection logic in `main.py` as needed
       (e.g. impose `n_estimators ≤ 120`).

4. **Reporting** – The script prints the best hyper‑parameters, their
   cross‑validated accuracy and the top‑10 most important TF‑IDF
   n‑grams.

---

## 2. Directory Structure

```

task2/
│
├── data/
│   └── data\_3.csv              # 500+ Arabic questions (sampled)
│
├── modules/
│   ├── DataLoader.py
│   ├── GeneticAlgorithm.py
│   ├── ParticleSwarmOptimizer.py
│   ├── RandomForestTuning.py   # single & multi‑objective fitness
│   ├── MultiObjectiveGA.py     # minimal NSGA‑II
│   ├── OptimizationProblem.py
│   ├── Individual.py
│   └── **init**.py
│
├── main.py                     # entry‑point
└── README.md                   # (this file)

````

---

## 3. Installation

```bash
# 1. clone / copy the repository
cd task2

# 2. (optional) create a virtual environment
python -m venv .venv
.\.venv\Scripts\activate         # Windows
# source .venv/bin/activate     # Linux / macOS

# 3. install dependencies
pip install -r requirements.txt
````

### `requirements.txt`

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.4
```

*(All algorithms rely only on NumPy, pandas and scikit‑learn.)*

---

## 4. Usage

### 4.1  Single‑objective optimisation

```bash
python main.py
```

Typical console output (truncated):

```
🔧  Multi‑objective mode: False

GA  best : [136, 11,  4,  2] | acc=0.884
PSO best : [128, 13,  2,  1] | acc=0.891

FINAL hyper‑params [128, 13,  2,  1]  → CV acc ≈ 0.891

Top‑10 TF‑IDF n‑grams:
طبيب                 0.0143
...
```

### 4.2  Multi‑objective optimisation

```bash
python main.py --multi
```

Example excerpt:

```
🔧  Multi‑objective mode: True

Pareto‑optimal set (accuracy ↑, size ↓):
  acc=0.888 | size= 70  ->  [ 70, 12,  3,  1]
  acc=0.890 | size=100  ->  [100, 14,  2,  2]
  acc=0.891 | size=128  ->  [128, 13,  2,  1]

✔  Selected: acc=0.891, size=128 => [128, 13,  2,  1]

FINAL hyper‑params [128, 13,  2,  1]  → CV acc ≈ 0.891
...
```

---

## 5. Module Guide

| Module                     | Purpose                                                                                            |
| -------------------------- | -------------------------------------------------------------------------------------------------- |
| **DataLoader**             | Reads CSV, cleans text, builds TF‑IDF matrix, exposes feature names.                               |
| **RandomForestTuning**     | Encapsulates 4 RF hyper‑parameters, single & multi‑objective evaluation (`fitness`, `objectives`). |
| **GeneticAlgorithm**       | Vanilla GA with tournament selection, uniform crossover & integer mutation.                        |
| **ParticleSwarmOptimizer** | Integer‑clamped PSO tailored to the same search space.                                             |
| **MultiObjectiveGA**       | Minimal NSGA‑II (fast non‑dominated sort + crowding distance).                                     |
| **main.py**                | CLI glue – data loading, optimisation pipeline, final training, reporting.                         |

---

## 6. Extending the Project

| Goal                                               | Where to Start                                                                                      |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **Add new hyper‑parameters** (e.g. `max_features`) | Append to `_param_info` in `RandomForestTuning.py`.                                                 |
| **Change fitness metric** (e.g. F1‑score)          | `cross_val_score(..., scoring="f1_weighted")` inside `RandomForestTuning.fitness`.                  |
| **Add more objectives**                            | Return extra values from `objectives()`, update `dominates()` in `MultiObjectiveGA.py` accordingly. |
| **Swap in another model**                          | Create a sibling class to `RandomForestTuning` that implements the same interface.                  |

---

## 7. Troubleshooting

| Symptom                                            | Cause & Fix                                                                                         |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `TypeError: cannot unpack non‑iterable int object` | `param_ranges` entry was overwritten – keep it as `(low, high)` tuples (fixed in the current code). |
| Poor accuracy (< 0.5)                              | Dataset too small or highly imbalanced – review `data/data_3.csv` sampling or increase `cv`.        |
| Script hangs / slow                                | Reduce `pop_size` or `generations` / `iterations`; use fewer CV folds.                              |

---

## 8. Dataset Citation

The sample data derives from **Altibbi Arabic medical Q/A** (public subset,
trimmed and shuffled for academic illustration).
Please ensure compliance with the original licence for any external use.

---

## 9. Licence

This repository is released under the **MIT Licence** – see `LICENSE` for
details.

---

*Happy Optimising!* 🚀

```
