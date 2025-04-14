import cma
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from solver import electro_thermal_timing_simulation

class OptimizationResult:
    """Class to store and analyze optimization results"""
    
    def __init__(self):
        self.evaluations = []
        self.best_params = None
        self.best_metrics = None
        self.best_cost = float('inf')
        self.start_time = time.time()
        
    def add_evaluation(self, params, metrics, cost):
        """Add an evaluation result"""
        self.evaluations.append({
            'params': params.copy(),
            'metrics': metrics.copy(),
            'cost': cost,
            'time': time.time() - self.start_time
        })
        
        if cost < self.best_cost:
            self.best_params = params.copy()
            self.best_metrics = metrics.copy()
            self.best_cost = cost
            
    def plot_convergence(self, output_dir="./optimization_results"):
        """Plot optimization convergence"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        costs = [e['cost'] for e in self.evaluations]
        times = [e['time'] for e in self.evaluations]
        
        # Plot cost vs evaluation
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(costs)), costs, 'o-')
        plt.xlabel('Evaluation')
        plt.ylabel('Cost')
        plt.title('Optimization Convergence')
        plt.grid(True)
        plt.savefig(f"{output_dir}/convergence_evals.png")
        
        # Plot cost vs time
        plt.figure(figsize=(10, 6))
        plt.plot(times, costs, 'o-')
        plt.xlabel('Time (s)')
        plt.ylabel('Cost')
        plt.title('Optimization Convergence over Time')
        plt.grid(True)
        plt.savefig(f"{output_dir}/convergence_time.png")
        
        # Plot parameter evolution
        param_names = [f"param_{i}" for i in range(len(self.evaluations[0]['params']))]
        param_values = {name: [] for name in param_names}
        
        for e in self.evaluations:
            for i, p in enumerate(e['params']):
                param_values[param_names[i]].append(p)
                
        plt.figure(figsize=(12, 8))
        for name, values in param_values.items():
            plt.plot(range(len(values)), values, 'o-', label=name)
        plt.xlabel('Evaluation')
        plt.ylabel('Parameter Value')
        plt.title('Parameter Evolution')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/parameter_evolution.png")
        
        # Save best results to file
        with open(f"{output_dir}/best_results.txt", 'w') as f:
            f.write("Best Parameters:\n")
            for i, p in enumerate(self.best_params):
                f.write(f"{param_names[i]}: {p}\n")
            f.write("\nBest Metrics:\n")
            for k, v in self.best_metrics.items():
                f.write(f"{k}: {v}\n")
            f.write(f"\nBest Cost: {self.best_cost}\n")
            f.write(f"Total Evaluations: {len(self.evaluations)}\n")
            f.write(f"Total Time: {times[-1]} seconds\n")
            
        print(f"Optimization results saved to {output_dir}/")
        
# Global optimization results object
opt_results = OptimizationResult()

def evaluate_design(params):
    """
    Evaluate a design configuration and return the cost
    
    Parameters:
    -----------
    params : list
        List of design parameters:
        [0] V_boundary - Applied voltage (V)
        [1] k_0 - Thermal conductivity (W/m-K)
        [2] nx - Number of elements in x direction
        [3] Lx - Physical length in x direction (mm)
        [4] clock_frequency - Clock frequency (GHz)
    
    Returns:
    --------
    float
        Cost value (lower is better)
    """
    # Unpack parameters
    V_boundary = params[0]
    k_0 = params[1]
    nx = int(params[2])
    ny = nx  # Keep mesh elements uniform
    nz = nx  # Keep mesh elements uniform
    Lx = params[3] / 1000.0  # Convert from mm to m
    Ly = Lx  # Square base
    Lz = Lx * 0.2  # Typical IC aspect ratio (height = 20% of length)
    clock_frequency = params[4] * 1e9  # Convert from GHz to Hz
    
    if nx > 30:  # Limit to 15³ elements maximum
        print(f"⚠️ Limiting mesh resolution from {nx} to 15 to prevent memory issues")
        nx = 15
        params[2] = 15.0

    approx_dofs = nx**3 * 3  # 3 unknowns per node
    approx_memory_gb = approx_dofs * approx_dofs * 8 / (1024**3)  # rough estimate
    
    if approx_memory_gb > 16:  # Adjust based on your system's RAM
        print(f"⚠️ Estimated memory usage too high ({approx_memory_gb:.1f} GB). Reducing mesh.")
        while approx_memory_gb > 16 and nx > 15:
            nx -= 5
            params[2] = float(nx)
            approx_dofs = nx**3 * 3
            approx_memory_gb = approx_dofs * approx_dofs * 8 / (1024**3)
        print(f"Adjusted mesh resolution to {nx}")

    # Print current evaluation parameters
    print(f"\nEvaluating: V={V_boundary:.2f}V, k={k_0:.2f}W/m-K, mesh={nx}x{ny}x{nz}, " 
          f"L={Lx*1000:.2f}mm, f={clock_frequency/1e9:.2f}GHz")
    
    try:
        # Start timer for this evaluation
        eval_start_time = time.time()
        
        # Run simulation
        results = electro_thermal_timing_simulation(
            nx=nx, ny=ny, nz=nz,
            Lx=Lx, Ly=Ly, Lz=Lz,
            V_boundary=V_boundary,
            k_0=k_0,
            clock_frequency=clock_frequency,
            max_iterations=30,
            tolerance=1e-7
        )
        
        # Extract metrics
        max_temp = results['max_temperature']
        temp_range = results['temp_range']
        max_gradient = results['max_thermal_gradient']
        clock_skew = results['clock_skew']
        cpu_time = results['cpu_time']
        
        # Create normalized metrics (lower is better)
        norm_temp = max_temp / 350.0        # Normalize to 350K reference temperature
        norm_gradient = max_gradient / 1000.0  # Normalize to 1000K/m reference gradient
        norm_skew = clock_skew * 1e11       # Convert to 0.1ns scale
        norm_time = cpu_time / 60.0         # Normalize to minutes
        
        # Define weights for multi-objective optimization
        w_temp = 0.25       # Weight for temperature
        w_gradient = 0.35   # Weight for thermal gradient (most important)
        w_skew = 0.30       # Weight for clock skew
        w_time = 0.10       # Weight for CPU time (least important)
        
        # Calculate weighted cost (lower is better)
        cost = (
            w_temp * norm_temp +
            w_gradient * norm_gradient +
            w_skew * norm_skew +
            w_time * norm_time
        )
        
        # Create metrics dictionary for tracking
        metrics = {
            'max_temperature': max_temp,
            'temperature_range': temp_range,
            'max_thermal_gradient': max_gradient,
            'clock_skew': clock_skew,
            'cpu_time': cpu_time,
            'normalized_cost': cost,
            'evaluation_time': time.time() - eval_start_time
        }
        
        # Add to optimization results
        opt_results.add_evaluation(params, metrics, cost)
        
        # Print results
        print(f"Results: MaxT={max_temp:.1f}K, Gradient={max_gradient:.1f}K/m, "
              f"Skew={clock_skew*1e12:.1f}ps, Time={cpu_time:.1f}s, Cost={cost:.4f}")
        
        return cost
        
    except Exception as e:
        print(f"❌ Error with parameters {params}: {str(e)}")
        return 1e6  # High cost for failed evaluations

def parallel_evaluate(solutions):
    """Evaluate multiple solutions in parallel"""
    with multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count()-1)) as pool:
        costs = pool.map(evaluate_design, solutions)
    return costs

def optimize_design(
    max_iterations=20, 
    population_size=8,
    output_dir="./optimization_results"
):
    """
    Optimize 3D IC design using CMA-ES
    
    Parameters:
    -----------
    max_iterations : int
        Maximum number of generations for CMA-ES
    population_size : int
        Population size per generation
    output_dir : str
        Directory to save results
    
    Returns:
    --------
    dict
        Optimization results
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initial design parameters
    # Format: [V_boundary, k_0, nx, Lx (mm), clock_frequency (GHz)]
    x0 = [10.0, 1.0, 5, 5.0, 1.0]
    
    # Parameter ranges (min, max)
    # Format: [[min_V, min_k, min_nx, min_Lx, min_f], [max_V, max_k, max_nx, max_Lx, max_f]]
    bounds = [
        [5.0, 0.5, 5.0, 1.0, 0.5],   # Lower bounds - reduce min mesh size
        [20.0, 5.0, 15.0, 10.0, 3.0]  # Upper bounds - reduce max mesh size
    ]
    
    # Initial step sizes (sigma)
    sigma0 = 0.3 * (np.array(bounds[1]) - np.array(bounds[0]))
    
    # CMA-ES options
    opts = {
        'bounds': bounds,
        'maxiter': max_iterations,
        'popsize': population_size,
        'CMA_diagonal': True,  # Diagonal covariance matrix (faster for fewer parameters)
        'verb_disp': 1,        # Display level
        'verb_log': 0,         # Log level
        'seed': 42             # Random seed for reproducibility
    }
    
    print(f"\n{'='*60}")
    print(f"Starting CMA-ES optimization with {population_size} designs per generation")
    print(f"Maximum {max_iterations} generations")
    print(f"Parameter space: {bounds[0]} to {bounds[1]}")
    print(f"{'='*60}\n")
    
    # Initialize CMA-ES
    es = cma.CMAEvolutionStrategy(x0, sigma0[0], opts)
    
    try:
        # Run optimization
        while not es.stop():
            # Get candidate solutions
            solutions = es.ask()
            
            # Evaluate solutions in parallel
            costs = parallel_evaluate(solutions)
            
            # Update CMA-ES with results
            es.tell(solutions, costs)
            
            # Display progress
            es.disp()
            
            # Optional early stopping
            if min(costs) < 0.8:  # If we found a very good solution
                print("Found excellent solution, stopping early.")
                break
        
        # Get best solution
        result = es.result
        best_params = result.xbest
        best_cost = result.fbest
        
        print("\n✅ Optimization Complete")
        print(f"Best Parameters:")
        print(f"  Voltage: {best_params[0]:.2f} V")
        print(f"  Thermal Conductivity: {best_params[1]:.2f} W/m-K")
        print(f"  Mesh Resolution: {int(best_params[2])}x{int(best_params[2])}x{int(best_params[2])}")
        print(f"  IC Size: {best_params[3]:.2f} mm")
        print(f"  Clock Frequency: {best_params[4]:.2f} GHz")
        print(f"Best Cost: {best_cost:.6f}")
        
        # Plot results
        # opt_results.plot_convergence(output_dir)
        
        # Run final simulation with best parameters for detailed analysis
        print("\nRunning final simulation with best parameters...")
        final_result = evaluate_design(best_params)
        
        return {
            'best_params': best_params,
            'best_cost': best_cost,
            'es_result': result,
            'optimization_history': opt_results
        }
        
    except KeyboardInterrupt:
        print("\n⚠️ Optimization interrupted by user")
        # Plot results even if interrupted
        opt_results.plot_convergence(output_dir)
        return {
            'best_params': opt_results.best_params,
            'best_cost': opt_results.best_cost,
            'interrupted': True,
            'optimization_history': opt_results
        }

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    optimize_design(max_iterations=10, population_size=6)