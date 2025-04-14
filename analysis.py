import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from solver import electro_thermal_timing_simulation, plot_results
import time

def analyze_parameter_sensitivity(
    baseline_params={
        'V_boundary': 10.0,
        'k_0': 1.0,
        'nx': 30,
        'Lx': 0.005,  # 5mm
        'clock_frequency': 1e9  # 1 GHz
    },
    output_dir="./sensitivity_analysis"
):
    """
    Perform sensitivity analysis by varying each parameter individually
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Parameters to vary and their ranges
    param_ranges = {
        'V_boundary': np.linspace(5.0, 10.0, 1),
        'k_0': np.linspace(0.5, 5.0, 5),
        'nx': np.array([20, 25, 30, 35, 40], dtype=int),
        'Lx': np.linspace(0.001, 0.01, 5),  # 1mm to 10mm
        'clock_frequency': np.linspace(0.5e9, 3e9, 5)  # 0.5 GHz to 3 GHz
    }
    
    # Metrics to track
    metrics = ['max_temperature', 'max_thermal_gradient', 'clock_skew', 'cpu_time']
    
    # Results storage
    results = {param: {metric: [] for metric in metrics} for param in param_ranges}
    
    # Run baseline simulation
    print(f"Running baseline simulation with parameters:")
    for param, value in baseline_params.items():
        print(f"  {param}: {value}")
        
    baseline_result = electro_thermal_timing_simulation(
        nx=baseline_params['nx'],
        ny=baseline_params['nx'],
        nz=baseline_params['nx'],
        Lx=baseline_params['Lx'],
        Ly=baseline_params['Lx'],
        Lz=baseline_params['Lx'] * 0.2,
        V_boundary=baseline_params['V_boundary'],
        k_0=baseline_params['k_0'],
        clock_frequency=baseline_params['clock_frequency']
    )
    
    # Plot baseline results
    plot_results(baseline_result, output_dir=f"{output_dir}/baseline")
    
    # Perform sensitivity analysis for each parameter
    for param, values in param_ranges.items():
        print(f"\n{'='*60}")
        print(f"Sensitivity analysis for parameter: {param}")
        print(f"{'='*60}")
        
        for value in values:
            # Skip baseline case to avoid duplicate simulation
            if np.isclose(value, baseline_params[param]):
                for metric in metrics:
                    results[param][metric].append(baseline_result[metric])
                continue
                
            print(f"\nTesting {param} = {value}")
            
            # Create parameter set with this value changed
            params = baseline_params.copy()
            params[param] = value
            
            # Run simulation
            try:
                start_time = time.time()
                result = electro_thermal_timing_simulation(
                    nx=params['nx'],
                    ny=params['nx'],
                    nz=params['nx'],
                    Lx=params['Lx'],
                    Ly=params['Lx'],
                    Lz=params['Lx'] * 0.2,
                    V_boundary=params['V_boundary'],
                    k_0=params['k_0'],
                    clock_frequency=params['clock_frequency']
                )
                
                # Store results
                for metric in metrics:
                    results[param][metric].append(result[metric])
                    
                print(f"Results: MaxT={result['max_temperature']:.1f}K, " 
                      f"Gradient={result['max_thermal_gradient']:.1f}K/m, "
                      f"Skew={result['clock_skew']*1e12:.1f}ps, "
                      f"Time={result['cpu_time']:.1f}s")
                      
            except Exception as e:
                print(f"❌ Error with {param} = {value}: {str(e)}")
                # Append NaN for failed simulations
                for metric in metrics:
                    results[param][metric].append(np.nan)
    
    # Plot sensitivity results
    plt.figure(figsize=(18, 12))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        
        for param, values in param_ranges.items():
            # Normalize values for comparison
            norm_values = values / baseline_params[param]
            metric_values = np.array(results[param][metric])
            
            if metric == 'cpu_time':
                # For CPU time, lower is better, so we invert
                norm_metric = baseline_result[metric] / metric_values
            else:
                # For other metrics, lower is also better
                norm_metric = baseline_result[metric] / metric_values
                
            plt.plot(norm_values, norm_metric, 'o-', label=param)
            
        plt.grid(True)
        plt.xlabel('Normalized Parameter Value')
        plt.ylabel('Normalized Performance (higher is better)')
        plt.title(f'Sensitivity Analysis: {metric}')
        plt.legend()
        
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sensitivity_analysis.png")
    
    # Save raw data for further analysis
    np.save(f"{output_dir}/sensitivity_results.npy", results)
    
    print(f"\nSensitivity analysis complete. Results saved to {output_dir}/")
    return results

def run_pareto_analysis(
    output_dir="./pareto_analysis"
):
    """
    Run simulations to identify Pareto front between conflicting objectives
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Define parameter ranges for exploration
    V_range = np.linspace(5.0, 20.0, 4)
    k_range = np.linspace(0.5, 5.0, 4)
    nx_range = np.array([20, 30, 40])
    
    # Results storage
    results = []
    
    # Run simulations
    for V in V_range:
        for k in k_range:
            for nx in nx_range:
                params = {
                    'V_boundary': V,
                    'k_0': k,
                    'nx': nx,
                    'Lx': 0.005,  # Fixed
                    'clock_frequency': 1e9  # Fixed
                }
                
                print(f"\nRunning simulation with V={V:.1f}, k={k:.1f}, nx={nx}")
                
                try:
                    result = electro_thermal_timing_simulation(
                        nx=params['nx'],
                        ny=params['nx'],
                        nz=params['nx'],
                        Lx=params['Lx'],
                        Ly=params['Lx'],
                        Lz=params['Lx'] * 0.2,
                        V_boundary=params['V_boundary'],
                        k_0=params['k_0'],
                        clock_frequency=params['clock_frequency']
                    )
                    
                    # Store relevant metrics
                    result_entry = {
                        'params': params,
                        'thermal_gradient': result['max_thermal_gradient'],
                        'clock_skew': result['clock_skew'],
                        'cpu_time': result['cpu_time']
                    }
                    
                    results.append(result_entry)
                    
                    print(f"Results: Gradient={result['max_thermal_gradient']:.1f}K/m, "
                          f"Skew={result['clock_skew']*1e12:.1f}ps, "
                          f"Time={result['cpu_time']:.1f}s")
                          
                except Exception as e:
                    print(f"❌ Error with parameters: {str(e)}")
    
    # Convert to numpy arrays for easier manipulation
    gradients = np.array([r['thermal_gradient'] for r in results])
    skews = np.array([r['clock_skew'] for r in results])
    times = np.array([r['cpu_time'] for r in results])
    
    # Plot Pareto fronts
    plt.figure(figsize=(12, 10))
    
    # Thermal gradient vs clock skew
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(gradients, skews*1e12, c=times, cmap='viridis')
    plt.colorbar(scatter, label='CPU Time (s)')
    plt.xlabel('Thermal Gradient (K/m)')
    plt.ylabel('Clock Skew (ps)')
    plt.title('Thermal Gradient vs Clock Skew')
    plt.grid(True)
    
    # Find and mark Pareto optimal points (gradient vs skew)
    pareto_points = []
    for i in range(len(results)):
        dominated = False
        for j in range(len(results)):
            if i != j:
                if (gradients[j] <= gradients[i] and skews[j] <= skews[i] and 
                    (gradients[j] < gradients[i] or skews[j] < skews[i])):
                    dominated = True
                    break
        if not dominated:
            pareto_points.append(i)
            plt.plot(gradients[i], skews[i]*1e12, 'ro', markersize=10)
    
    # CPU time vs thermal gradient
    plt.subplot(2, 2, 2)
    scatter = plt.scatter(times, gradients, c=skews*1e12, cmap='viridis')
    plt.colorbar(scatter, label='Clock Skew (ps)')
    plt.xlabel('CPU Time (s)')
    plt.ylabel('Thermal Gradient (K/m)')
    plt.title('CPU Time vs Thermal Gradient')
    plt.grid(True)
    
    # CPU time vs clock skew
    plt.subplot(2, 2, 3)
    scatter = plt.scatter(times, skews*1e12, c=gradients, cmap='viridis')
    plt.colorbar(scatter, label='Thermal Gradient (K/m)')
    plt.xlabel('CPU Time (s)')
    plt.ylabel('Clock Skew (ps)')
    plt.title('CPU Time vs Clock Skew')
    plt.grid(True)
    
    # 3D plot of all three metrics
    ax = plt.subplot(2, 2, 4, projection='3d')
    scatter = ax.scatter(gradients, skews*1e12, times, c=times, cmap='viridis')
    plt.colorbar(scatter, label='CPU Time (s)')
    ax.set_xlabel('Thermal Gradient (K/m)')
    ax.set_ylabel('Clock Skew (ps)')
    ax.set_zlabel('CPU Time (s)')
    ax.set_title('3D Trade-off Space')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pareto_analysis.png")
    
    # Save Pareto optimal configurations
    with open(f"{output_dir}/pareto_optimal.txt", "w") as f:
        f.write("Pareto Optimal Configurations (Thermal Gradient vs Clock Skew):\n\n")
        for i in pareto_points:
            f.write(f"Configuration {i+1}:\n")
            f.write(f"  Parameters:\n")
            for param, value in results[i]['params'].items():
                f.write(f"    {param}: {value}\n")
            f.write(f"  Results:\n")
            f.write(f"    Thermal Gradient: {gradients[i]:.2f} K/m\n")
            f.write(f"    Clock Skew: {skews[i]*1e12:.2f} ps\n")
            f.write(f"    CPU Time: {times[i]:.2f} s\n\n")
    
    # Save raw data
    np.save(f"{output_dir}/pareto_results.npy", results)
    
    print(f"\nPareto analysis complete. Results saved to {output_dir}/")
    return results

if __name__ == "__main__":
    print("\n1. Running parameter sensitivity analysis...")
    sensitivity_results = analyze_parameter_sensitivity()
    
    print("\n2. Running Pareto front analysis...")
    pareto_results = run_pareto_analysis()