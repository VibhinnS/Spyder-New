from cma import CMAEvolutionStrategy
import numpy as np

# Define the objective function to be optimized
def objective_function(params, solver_params):
    """
    Objective function for CMA-ES optimization.
    
    Parameters:
    -----------
    params : list or np.array
        Optimization parameters: [clock_frequency, wire_resistivity, wire_capacitance]
        
    solver_params : dict
        Parameters passed to the electro-thermal solver
    
    Returns:
    --------
    float
        The weighted combination of thermal gradient, CPU time, and clock skew to minimize
    """
    # Unpack parameters
    clock_frequency = params[0]
    wire_resistivity = params[1]
    wire_capacitance = params[2]
    
    # Update solver parameters
    solver_params['clock_frequency'] = clock_frequency
    solver_params['wire_resistivity'] = wire_resistivity
    solver_params['wire_capacitance'] = wire_capacitance
    
    # Run the electro-thermal solver with updated parameters
    results = electro_thermal_timing_simulation(**solver_params)
    
    # Extract desired metrics from the results
    thermal_gradient = results['max_thermal_gradient']
    cpu_time = results['cpu_time']
    clock_skew = results['clock_skew']
    
    # Normalize the objectives for weighting (these can be adjusted based on desired importance)
    # We want to minimize these, so lower values are better
    weight_thermal_gradient = 0.5
    weight_cpu_time = 0.3
    weight_clock_skew = 0.2
    
    # Combine the objectives into a single cost (to minimize)
    cost = weight_thermal_gradient * thermal_gradient + \
           weight_cpu_time * cpu_time + \
           weight_clock_skew * clock_skew
    
    return cost

def run_optimization(initial_params, solver_params, options=None):
    """
    Run CMA-ES optimization for the electro-thermal simulation.
    
    Parameters:
    -----------
    initial_params : list or np.array
        Initial guess for the optimization parameters: [clock_frequency, wire_resistivity, wire_capacitance]
        
    solver_params : dict
        Parameters passed to the electro-thermal solver
    
    options : dict
        CMA-ES options (optional)
        
    Returns:
    --------
    dict
        Optimization results, including optimized parameters and the objective function history
    """
    # Set default options for CMA-ES if not provided
    if options is None:
        options = {
            'popsize': 20,        # Population size for CMA-ES
            'maxiter': 50,        # Maximum number of generations
            'tolx': 1e-5,         # Tolerance for convergence (in parameter space)
            'tolfun': 1e-5,       # Tolerance for convergence (in objective function space)
        }
    
    # Initialize the CMA-ES optimizer
    es = CMAEvolutionStrategy(initial_params, 0.5, options)  # 0.5 is the initial sigma (step size)
    
    # Optimization loop
    while not es.stop():
        # Get the current population (candidate solutions)
        solutions = es.ask()
        
        # Evaluate the objective function for each candidate solution
        fitness_values = []
        for solution in solutions:
            cost = objective_function(solution, solver_params)
            fitness_values.append(cost)
        
        # Tell the optimizer about the results
        es.tell(solutions, fitness_values)
        
        # Print the current progress
        print(f"Iteration {es.count}: Best fitness = {es.result[0]}")

    # Get the best solution found
    best_solution = es.result[0]
    best_fitness = es.result[1]
    
    # Return the optimization results
    return {
        'best_solution': best_solution,
        'best_fitness': best_fitness,
        'history': es.result[2]
    }

# Example usage:
# Initial parameter guess (clock_frequency, wire_resistivity, wire_capacitance)
initial_params = [1e9, 1.68e-8, 1e-15]

# Solver parameters for electro-thermal simulation (defaults, but can be overridden)
solver_params = {
    'nx': 10, 'ny': 10, 'nz': 10,
    'Lx': 1.0, 'Ly': 1.0, 'Lz': 1.0,
    'T_0': 300.0, 'V_boundary': 10.0, 'T_boundary': 300.0,
    'rho_0': 1.0, 'alpha': 0.004, 'k_0': 1.0,
    'clock_frequency': 1e9, 'wire_resistivity': 1.68e-8, 'wire_capacitance': 1e-15,
    'material_zones': None, 'max_iterations': 20, 'tolerance': 1e-8
}

# Run optimization
optimization_results = run_optimization(initial_params, solver_params)

# Print the results
print("Optimization complete.")
print(f"Best solution: {optimization_results['best_solution']}")
print(f"Best fitness (objective value): {optimization_results['best_fitness']}")
