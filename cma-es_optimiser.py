import cma
import multiprocessing
from solver import electro_thermal_simulation

def evaluate_design(params):
    V_boundary = params[0]

    try:
        results = electro_thermal_simulation(V_boundary=V_boundary)
        T = results['temperature']
        P = results['power_density']

        max_temp = T.vector().max()
        avg_temp = T.vector().sum() / T.vector().size()
        skew = max_temp - avg_temp

        cost = (
            max_temp +       # prioritize temperature
            0.5 * skew +     # add skew as secondary
            0.1 * V_boundary # prefer lower voltage
        )

        print(f"V={V_boundary:.2f} -> MaxT={max_temp:.2f}, Skew={skew:.2f}, Cost={cost:.2f}")
        return cost

    except Exception as e:
        print(f"\u274c Error with V={V_boundary}: {e}")
        return 1e6

def parallel_evaluate(solutions):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        costs = pool.map(evaluate_design, solutions)
    return costs

def optimize_design():
    x0 = [10.0]  # Initial guess
    sigma0 = 1.0
    bounds = [[5.0], [20.0]]

    es = cma.CMAEvolutionStrategy(
        x0, sigma0,
        {'bounds': bounds, 'maxiter': 10, 'popsize': 6}
    )

    while not es.stop():
        solutions = es.ask()
        costs = parallel_evaluate(solutions)
        es.tell(solutions, costs)
        es.disp()

    result = es.result
    print("\n\u2705 Optimization Complete")
    print(f"Best Voltage: V={result.xbest[0]:.2f}")
    print(f"Minimum Cost: {result.fbest:.3f}")
    return result

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    optimize_design()
