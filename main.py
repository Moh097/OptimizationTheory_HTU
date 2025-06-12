import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from modules import (
    DataLoader,
    GeneticAlgorithm,
    ParticleSwarmOptimizer,
    RandomForestTuning,
    MultiObjectiveGA,     
)


def build_final_model(best_params, X, y):
    """Train and return the final RandomForest with *best_params*."""
    return RandomForestClassifier(
        n_estimators=int(best_params[0]),
        max_depth=int(best_params[1]),
        min_samples_split=int(best_params[2]),
        min_samples_leaf=int(best_params[3]),
        random_state=42,
    ).fit(X, y)


def main():
    # ------------------------------------------------------------------ set‑up
    parser = argparse.ArgumentParser(description="Random‑Forest hyper‑parameter optimisation")
    parser.add_argument("--multi", action="store_true",
                        help="Activate multi‑objective mode (NSGA‑II)")
    args = parser.parse_args()

    print(f"🔧  Multi‑objective mode: {args.multi}\n")

    # ---------------------------------------------------------------- data
    loader = DataLoader("data/data.csv")
    X, y = loader.get_features_and_labels()
    X = X.toarray()

    problem = RandomForestTuning(X, y, cv=5)

    # ============================================================= optimisation
    if args.multi:
        # ---------- NSGA‑II (built from scratch, see modules/MultiObjectiveGA.py)
        nsga = MultiObjectiveGA(problem, pop_size=40, generations=30,
                                crossover_rate=0.8, mutation_rate=0.2)
        pareto = nsga.evolve()          # list of (params, (acc, size))
        print("Pareto‑optimal set (accuracy ↑, size ↓):")
        for params, (acc, size) in pareto:
            print(f"  acc={acc:.3f} | size={size:3d}  ->  {params}")

        # pick solution with *highest accuracy* from Pareto set for final model
        best_params, (best_acc, best_size) = max(pareto, key=lambda t: t[1][0])
        print(f"\n✔  Selected: acc={best_acc:.3f}, size={best_size} => {best_params}")

        final_model = build_final_model(best_params, X, y)

    else:
        # ---------- single‑objective (unchanged pipeline)
        ga = GeneticAlgorithm(problem, pop_size=30, generations=20,
                              crossover_rate=0.8, mutation_rate=0.2)
        best_ga, score_ga = ga.evolve()
        print(f"GA  best : {best_ga} | acc={score_ga:.3f}")

        pso = ParticleSwarmOptimizer(problem, pop_size=30,
                                     w=0.5, c1=1.0, c2=1.0, iterations=20)
        best_pso, score_pso = pso.optimize()
        print(f"PSO best : {best_pso} | acc={score_pso:.3f}")

        best_params = best_ga if score_ga >= score_pso else best_pso
        best_acc = max(score_ga, score_pso)

        final_model = build_final_model(best_params, X, y)

    # ============================================================= reporting
    print(f"\nFINAL hyper‑params {best_params}  → CV acc ≈ {best_acc:.3f}")

    # ---------- top features
    fi = final_model.feature_importances_
    names = loader.get_feature_names()
    top = sorted(zip(fi, names), reverse=True)[:10]
    print("\nTop‑10 TF‑IDF n‑grams:")
    for imp, n in top:
        print(f"{n:25s}  {imp:.4f}")


if __name__ == "__main__":
    main()
