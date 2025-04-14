#!/usr/bin/env python3
"""
3D IC Optimization Suite

This script provides an interface to run different components of the 3D IC
optimization system:
1. Single simulation with specified parameters
2. Parameter sweep for sensitivity analysis
3. Full optimization using CMA-ES
"""

import argparse
import os
import time
import multiprocessing
from solver import electro_thermal_timing_simulation, plot_results
from cma_es_optimiser import optimize_design
from analysis import analyze_parameter_sensitivity, run_pareto_analysis

def main():
    parser = argparse.ArgumentParser(description="3D IC Electro-Thermal-Timing Optimization")
    
    # Create subparsers for different functions
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Single simulation parser
    sim_parser = subparsers.add_parser("simulate", help="Run a single simulation with specified parameters")
    sim_parser.add_argument("--voltage", type=float, default=10.0, help="Applied voltage (V)")
    sim_parser.add_argument("--conductivity", type=float, default=1.0, help="Thermal conductivity (W/m-K)")
    sim_parser.add_argument("--mesh", type=int, default=30, help="Mesh resolution (nx=ny=nz)")
    sim_parser.add_argument("--size", type=float, default=5.0, help="IC size in mm")
    sim_parser.add_argument("--frequency", type=float, default=1.0, help="Clock frequency in GHz")
    sim_parser.add_argument("--output", type=str, default="./simulation_results", help="Output directory")
    
    # Sensitivity analysis parser
    sens_parser = subparsers.add_parser("sensitivity", help="Run parameter sensitivity analysis")
    sens_parser.add_argument("--baseline-voltage", type=float, default=10.0, help="Baseline voltage (V)")
    sens_parser.add_argument("--baseline-conductivity", type=float, default=1.0, help="Baseline thermal conductivity (W/m-K)")
    sens_parser.add_argument("--baseline-mesh", type=int, default=30, help="Baseline mesh resolution")
    sens_parser.add_argument("--baseline-size", type=float, default=5.0, help="Baseline IC size in mm")
    sens_parser.add_argument("--baseline-frequency", type=float, default=1.0, help="Baseline clock frequency in GHz")
    sens_parser.add_argument("--output", type=str, default="./sensitivity_results", help="Output directory")
    
    # Optimization parser
    opt_parser = subparsers.add_parser("optimize", help="Run CMA-ES optimization")
    opt_parser.add_argument("--generations", type=int, default=15, help="Maximum number of generations")
    opt_parser.add_argument("--population", type=int, default=6, help="Population size per generation")
    opt_parser.add_argument("--output", type=str, default="./optimization_results", help="Output directory")
    
    # Pareto analysis parser
    pareto_parser = subparsers.add_parser("pareto", help="Run Pareto front analysis")
    pareto_parser.add_argument("--output", type=str, default="./pareto_results", help="Output directory")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    
    # Execute the selected command
    if args.command == "simulate":
        print(f"\n{'='*60}")
        print(f"Running single simulation with specified parameters")
        print(f"{'='*60}")
        
        results = electro_thermal_timing_simulation(
            nx=args.mesh,
            ny=args.mesh,
            nz=args.mesh,
            Lx=args.size/1000.0,  # Convert mm to m
            Ly=args.size/1000.0,
            Lz=args.size*0.2/1000.0,  # Height is 20% of width
            V_boundary=args.voltage,
            k_0=args.conductivity,
            clock_frequency=args.frequency*1e9  # Convert GHz to Hz
        )
        
        # Print results
        print("\nSimulation Results:")
        print(f"  Maximum Temperature: {results['max_temperature']:.2f} K")
        print(f"  Temperature Range: {results['temp_range']:.2f} K")
        print(f"  Maximum Thermal Gradient: {results['max_thermal_gradient']:.2f} K/m")
        print(f"  Clock Skew: {results['clock_skew']*1e12:.2f} ps")
        print(f"  CPU Time: {results['cpu_time']:.2f} s")
        
        # Plot results
        plot_results(results, output_dir=args.output)
        
    elif args.command == "sensitivity":
        print(f"\n{'='*60}")
        print(f"Running parameter sensitivity analysis")
        print(f"{'='*60}")
        
        # Set up baseline parameters
        baseline_params = {
            'V_boundary': args.baseline_voltage,
            'k_0': args.baseline_conductivity,
            'nx': args.baseline_mesh,
            'Lx': args.baseline_size/1000.0,  # Convert mm to m
            'clock_frequency': args.baseline_frequency*1e9  # Convert GHz to Hz
        }
        
        # Run sensitivity analysis
        analyze_parameter_sensitivity(
            baseline_params=baseline_params,
            output_dir=args.output
        )
        
    elif args.command == "optimize":
        print(f"\n{'='*60}")
        print(f"Running CMA-ES optimization")
        print(f"{'='*60}")
        
        # Run optimization
        optimize_design(
            max_iterations=args.generations,
            population_size=args.population,
            output_dir=args.output
        )
        
    elif args.command == "pareto":
        print(f"\n{'='*60}")
        print(f"Running Pareto front analysis")
        print(f"{'='*60}")
        
        # Run Pareto analysis
        run_pareto_analysis(output_dir=args.output)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
        print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")