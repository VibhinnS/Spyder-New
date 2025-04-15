import csv
import numpy as np
from cma_es_optimiser import optimize_design as run_cma_es
from bo import optimize_design as run_bayesian_opt
def generate_random_parameters(n_samples=50000):
    bounds = [
        (5.0, 20.0),    # V_boundary
        (0.5, 5.0),     # k_0
        (1.0, 10.0),    # Lx
        (0.5, 3.0)      # clock_frequency
    ]
    return [
        [np.random.uniform(low, high) for (low, high) in bounds]
        for _ in range(n_samples)
    ]

def main():
    param_sets = generate_random_parameters()

    with open("optimization_results.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "V_boundary", "k_0", "Lx_mm", "clock_freq_GHz",
            "cma_es_iterations", "bo_iterations"
        ])
        
        for idx, param in enumerate(param_sets):
            print(f"\n=== Running set {idx+1}/{len(param_sets)} ===")

            # CMA-ES optimization
            try:
                result_cma = run_cma_es(
                    max_iterations=10000,  # <-- make this integer
                    population_size=4,
                    output_dir=f"./logs/cma_es_{idx}"
                )
                cma_iters = len(result_cma["optimization_history"].evaluations)
            except Exception as e:
                print(f"CMA-ES failed: {e}")
                cma_iters = -1

            # Bayesian Optimization
            try:
                result_bo = run_bayesian_opt(
                    max_evaluations=10000,  # <-- make this integer
                    output_dir=f"./logs/bo_{idx}"
                )
                bo_iters = len(result_bo["optimization_history"].evaluations)
            except Exception as e:
                print(f"Bayesian Optimization failed: {e}")
                bo_iters = -1

            # Write to CSV
            print('\n')
            print("Params - ", param)
            print("Params Type - ", type(param))
            print("CMA Iters - ", cma_iters)
            print("CMA Iters Type - ", type(cma_iters))
            print("BO Iters - ", bo_iters)
            print("BO Iters Type - ", type(bo_iters))
            print('\n')
            writer.writerow([
                *param, cma_iters, bo_iters
            ])

            print("Written for a bound set")

if __name__ == "__main__":
    main()
