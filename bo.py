import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from solver import electro_thermal_timing_simulation
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize
from scipy.stats import norm

class OptimizationResult:
    """Class to store and analyze optimization results"""
    
    def __init__(self):
        self.reset()

    def reset(self):
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
        param_names = ['Voltage (V)', 'Thermal Conductivity (W/m-K)', 
                       'IC Size (mm)', 'Clock Frequency (GHz)']
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
    params : list or numpy.ndarray
        List of design parameters:
        [0] V_boundary - Applied voltage (V)
        [1] k_0 - Thermal conductivity (W/m-K)
        [2] Lx - Physical length in x direction (mm)
        [3] clock_frequency - Clock frequency (GHz)
    
    Returns:
    --------
    float
        Cost value (lower is better)
    """
    # Convert to list if numpy array
    params = params.tolist() if isinstance(params, np.ndarray) else params
    
    # Unpack parameters
    V_boundary = params[0]
    k_0 = params[1]
    Lx = params[2] / 1000.0  # Convert from mm to m
    clock_frequency = params[3] * 1e9  # Convert from GHz to Hz
    
    # FIXED MESH SIZE - not part of optimization parameters
    nx = 20  # Fixed mesh size
    ny = nx  # Keep mesh elements uniform
    nz = nx  # Keep mesh elements uniform
    
    Ly = Lx  # Square base
    Lz = Lx * 0.2  # Typical IC aspect ratio (height = 20% of length)

    # Print current evaluation parameters
    print(f"\nEvaluating: V={V_boundary:.2f}V, k={k_0:.2f}W/m-K, mesh={nx}x{ny}x{nz} (fixed), " 
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

class BayesianOptimizer:
    """Bayesian Optimization implementation"""
    
    def __init__(self, bounds, n_init=5):
        """
        Initialize Bayesian Optimizer
        
        Parameters:
        -----------
        bounds : list of tuples
            Parameter bounds [(lower_1, upper_1), ..., (lower_n, upper_n)]
        n_init : int
            Number of initial random samples before using acquisition function
        """
        self.bounds = bounds
        self.dim = len(bounds)
        self.n_init = n_init
        self.X = []    # Evaluated parameters
        self.y = []    # Corresponding costs
        
        # Gaussian Process with Matern kernel (common choice for Bayesian optimization)
        # nu=2.5 is a good balance between smoothness and flexibility
        kernel = Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,  # Small regularization for numerical stability
            normalize_y=True,
            n_restarts_optimizer=5  # For GP hyperparameter optimization
        )
    
    def _acquisition_function(self, x, xi=0.01):
        """
        Expected Improvement acquisition function
        
        Parameters:
        -----------
        x : array-like
            Point to evaluate
        xi : float
            Exploration-exploitation trade-off parameter
            
        Returns:
        --------
        float
            Expected improvement at x
        """
        x = x.reshape(1, -1)
        
        # No model yet, return a small random value
        if not self.X:
            return np.random.random() * 0.01 - 0.005
        
        # Predict with the GP model
        mu, sigma = self.gp.predict(x, return_std=True)
        
        # Current best 
        y_best = np.min(self.y)
        
        # Calculate improvement
        imp = y_best - mu
        
        # Calculate Z-score for the CDF
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = imp / sigma
        
        # Expected improvement
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        # Apply exploration bonus
        ei = ei + xi * sigma
        
        return -ei  # Return negative because we minimize
    
    def _sample_next_point(self):
        """Sample next point based on acquisition function"""
        
        # If we don't have enough samples, choose randomly
        if len(self.X) < self.n_init:
            return self._random_sample()
        
        # Find the point that maximizes the acquisition function
        best_x = None
        best_acq = float('inf')
        
        # Try multiple starting points to avoid local minima
        n_restarts = 10
        bounds_array = np.array(self.bounds)
        
        # Start with random samples
        random_starts = []
        for _ in range(n_restarts):
            random_starts.append(self._random_sample())
            
        # Add the best point found so far as a starting point
        best_idx = np.argmin(self.y)
        random_starts.append(self.X[best_idx])
        
        # Run optimizer from each starting point
        for x_start in random_starts:
            result = minimize(
                self._acquisition_function,
                x_start,
                bounds=self.bounds,
                method='L-BFGS-B'
            )
            
            if result.fun < best_acq:
                best_acq = result.fun
                best_x = result.x
                
        return best_x
    
    def _random_sample(self):
        """Generate a random sample within bounds"""
        bounds_array = np.array(self.bounds)
        lower_bounds = bounds_array[:, 0]
        upper_bounds = bounds_array[:, 1]
        return lower_bounds + (upper_bounds - lower_bounds) * np.random.random(self.dim)
    
    def update_model(self):
        """Update the Gaussian Process model with available data"""
        if len(self.X) >= 2:  # Need at least 2 points to fit
            X_array = np.array(self.X)
            y_array = np.array(self.y)
            self.gp.fit(X_array, y_array)
    
    def suggest(self):
        """Suggest next evaluation point"""
        return self._sample_next_point()
    
    def register(self, x, y):
        """Register an evaluation result"""
        self.X.append(x)
        self.y.append(y)

def optimize_design(
    max_evaluations=10000,
    output_dir="./optimization_results"
):
    """
    Optimize 3D IC design using Bayesian Optimization
    
    Parameters:
    -----------
    max_evaluations : int
        Maximum number of function evaluations
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
    
    # Parameter bounds
    # Format: [(min_V, max_V), (min_k, max_k), (min_Lx, max_Lx), (min_f, max_f)]
    bounds = [
        (5.0, 20.0),    # V_boundary
        (0.5, 5.0),     # k_0
        (1.0, 10.0),    # Lx (mm)
        (0.5, 3.0)      # clock_frequency (GHz)
    ]
    
    print(f"\n{'='*60}")
    print(f"Starting Bayesian Optimization")
    print(f"Maximum {max_evaluations} evaluations")
    print(f"Parameter bounds:")
    param_names = ["Voltage (V)", "Thermal Conductivity (W/m-K)", 
                   "IC Size (mm)", "Clock Frequency (GHz)"]
    for i, (name, bound) in enumerate(zip(param_names, bounds)):
        print(f"  {name}: {bound[0]} to {bound[1]}")
    print(f"Using fixed mesh size: 10x10x10")
    print(f"{'='*60}\n")
    
    # Initialize Bayesian Optimizer
    n_init = min(5, max_evaluations // 3)  # Use ~1/3 of budget for initial exploration
    optimizer = BayesianOptimizer(bounds, n_init=n_init)
    
    try:
        # Main optimization loop
        for i in range(max_evaluations):
            print(f"\n[Iteration {i+1}/{max_evaluations}]")
            
            # Get next point to evaluate
            next_params = optimizer.suggest()
            
            # Evaluate the design
            cost = evaluate_design(next_params)
            
            # Register result
            optimizer.register(next_params, cost)
            
            # Update model
            optimizer.update_model()
            
            # Optional early stopping
            if cost < 0.8:  # If we found a very good solution
                print("Found excellent solution, stopping early.")
                break
        
        # Get best solution
        best_idx = np.argmin(optimizer.y)
        best_params = optimizer.X[best_idx]
        best_cost = optimizer.y[best_idx]
        
        print("\n✅ Optimization Complete")
        print(f"Best Parameters:")
        print(f"  Voltage: {best_params[0]:.2f} V")
        print(f"  Thermal Conductivity: {best_params[1]:.2f} W/m-K")
        print(f"  IC Size: {best_params[2]:.2f} mm")
        print(f"  Clock Frequency: {best_params[3]:.2f} GHz")
        print(f"  Mesh Size: 10x10x10 (fixed)")
        print(f"Best Cost: {best_cost:.6f}")
        
        # Plot results
        opt_results.plot_convergence(output_dir)
        
        # Run final simulation with best parameters for detailed analysis
        print("\nRunning final simulation with best parameters...")
        final_result = evaluate_design(best_params)
        
        return {
            'best_params': best_params,
            'best_cost': best_cost,
            'X': optimizer.X,
            'y': optimizer.y,
            'optimization_history': opt_results
        }
        
    except KeyboardInterrupt:
        print("\n⚠️ Optimization interrupted by user")
        # Plot results even if interrupted
        opt_results.plot_convergence(output_dir)
        
        # Get best solution so far
        if optimizer.y:
            best_idx = np.argmin(optimizer.y)
            best_params = optimizer.X[best_idx]
            best_cost = optimizer.y[best_idx]
        else:
            best_params = None
            best_cost = None
            
        return {
            'best_params': best_params,
            'best_cost': best_cost,
            'X': optimizer.X,
            'y': optimizer.y,
            'interrupted': True,
            'optimization_history': opt_results
        }

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    optimize_design(max_evaluations=20)