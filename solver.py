from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import time

parameters["linear_algebra_backend"] = "PETSc"
parameters["krylov_solver"]["absolute_tolerance"] = 1e-6
parameters["krylov_solver"]["relative_tolerance"] = 1e-4
parameters["krylov_solver"]["maximum_iterations"] = 2000
parameters["form_compiler"]["cpp_optimize_flags"] = "-O2"
parameters["form_compiler"]["cache_dir"] = "clear"
parameters["krylov_solver"]["monitor_convergence"] = False
parameters["krylov_solver"]["nonzero_initial_guess"] = True

def electro_thermal_timing_simulation(
    # Mesh parameters
    nx=20, ny=20, nz=20,
    Lx=1.0, Ly=1.0, Lz=1.0,
    
    # Physical parameters
    T_0=300.0,
    V_boundary=10.0,
    T_boundary=323.15,
    rho_0=1.0,
    alpha=0.004,
    k_0=1.0,
    
    # Clock parameters
    clock_frequency=1e9,  # 1 GHz
    wire_resistivity=1.68e-8,  # Copper resistivity (Ohm-m)
    wire_capacitance=1e-15,  # Wire capacitance per unit length (F/m)
    
    # Material parameters
    material_zones=None,  # Dict of subdomains with properties
    
    # Convergence parameters
    max_iterations=20,
    tolerance=1e-8,
):
    """
    Enhanced electro-thermal-timing simulation for 3D IC optimization.
    
    Parameters:
    -----------
    nx, ny, nz : int
        Number of elements in each dimension
    Lx, Ly, Lz : float
        Physical dimensions of the IC (m)
    T_0 : float
        Reference temperature (K)
    V_boundary : float
        Applied voltage (V)
    T_boundary : float
        Boundary temperature (K)
    rho_0 : float
        Base electrical resistivity (Ohm-m)
    alpha : float
        Temperature coefficient of resistivity (1/K)
    k_0 : float
        Thermal conductivity (W/m-K)
    clock_frequency : float
        Clock frequency (Hz)
    wire_resistivity : float
        Resistivity of clock wires (Ohm-m)
    wire_capacitance : float
        Capacitance of clock wires per unit length (F/m)
    material_zones : dict
        Dictionary defining material zones with properties
    max_iterations : int
        Maximum number of nonlinear iterations
    tolerance : float
        Convergence tolerance
        
    Returns:
    --------
    dict
        Simulation results
    """
    start_time = time.time()
    
    # Create mesh
    mesh = BoxMesh(Point(0, 0, 0), Point(Lx, Ly, Lz), nx, ny, nz)
    
    # Define function spaces
    V_space = FunctionSpace(mesh, 'P', 1)  # Voltage
    T_space = FunctionSpace(mesh, 'P', 1)  # Temperature
    C_space = FunctionSpace(mesh, 'P', 1)  # Clock signal delay
    
    # Define trial and test functions
    v = TrialFunction(V_space)
    w = TestFunction(V_space)
    t = TrialFunction(T_space)
    s = TestFunction(T_space)
    c = TrialFunction(C_space)
    d = TestFunction(C_space)
    
    # Define functions
    V = Function(V_space)         # Voltage
    T = Function(T_space)         # Temperature
    T_prev = Function(T_space)    # Previous temperature
    C = Function(C_space)         # Clock signal delay
    
    # Initialize temperature
    T_prev.interpolate(Constant(T_0))
    
    # Define material properties (can be updated based on material_zones)
    if material_zones is None:
        material_zones = {
            'default': {
                'resistivity': rho_0,
                'thermal_conductivity': k_0,
                'temp_coeff': alpha
            }
        }
        
    # Define boundary conditions
    def voltage_boundary(x, on_boundary):
        return on_boundary and near(x[0], 0, 1e-14)
    
    def ground_boundary(x, on_boundary):
        return on_boundary and near(x[0], Lx, 1e-14)
    
    def temperature_boundary(x, on_boundary):
        return on_boundary
    
    def clock_source_boundary(x, on_boundary):
        # Clock source at one corner
        return on_boundary and near(x[0], 0, 1e-14) and near(x[1], 0, 1e-14) and near(x[2], 0, 1e-14)
    
    # Apply boundary conditions
    bc_V_high = DirichletBC(V_space, Constant(V_boundary), voltage_boundary)
    bc_V_ground = DirichletBC(V_space, Constant(0.0), ground_boundary)
    bc_T = DirichletBC(T_space, Constant(T_boundary), temperature_boundary)
    bc_C = DirichletBC(C_space, Constant(0.0), clock_source_boundary)  # Zero delay at clock source
    
    bcs_V = [bc_V_high, bc_V_ground]
    bcs_T = [bc_T]
    bcs_C = [bc_C]
    
    # Nonlinear iteration loop for electro-thermal coupling
    iteration = 0
    error = 1.0
    errors = []
    
    print("Starting nonlinear electro-thermal iterations...")
    while iteration < max_iterations and error > tolerance:
        iteration += 1
        
        # Update resistivity based on temperature
        resistivity_expr = rho_0 * (1.0 + alpha * (T_prev - T_0))
        
        # Solve voltage equation
        a_V = (1.0 / resistivity_expr) * dot(grad(v), grad(w)) * dx
        L_V = Constant(0.0) * w * dx
        solve(a_V == L_V, V, bcs_V,
              solver_parameters={"linear_solver": "gmres",
                                 "preconditioner":"ilu"})
        
        # Calculate power density (Joule heating)
        V_grad = grad(V)
        power_density = (1.0 / resistivity_expr) * dot(V_grad, V_grad)
        
        # Solve temperature equation
        a_T = k_0 * dot(grad(t), grad(s)) * dx
        L_T = power_density * s * dx
        solve(a_T == L_T, T, bcs_T,
              solver_parameters={"linear_solver": "gmres", 
                                 "preconditioner": "ilu"})
        
        # Check convergence
        T_diff = T.copy(deepcopy=True)
        T_diff.vector().axpy(-1.0, T_prev.vector())
        error = norm(T_diff.vector()) / norm(T.vector())
        errors.append(error)
        
        print(f"Iteration {iteration}: error = {error:.6e}")
        T_prev.assign(T)
        
        if error < tolerance:
            print(f"✅ Converged after {iteration} iterations!")
            break
    
    if iteration == max_iterations and error > tolerance:
        print(f"⚠️ Warning: Did not converge after {max_iterations} iterations. Final error: {error:.6e}")
    
    # Calculate clock signal propagation delay based on temperature
    # RC delay model: delay ~ R*C, where R depends on temperature
    print("Calculating clock signal propagation delays...")
    
    # Temperature-dependent wire resistivity
    wire_res_T = wire_resistivity * (1.0 + alpha * (T - T_0))
    
    # Solve for clock signal delay
    a_C = inner(grad(c), grad(d)) * dx
    L_C = wire_res_T * wire_capacitance * d * dx
    
    solve(a_C == L_C, C, bcs_C)
    
    # Calculate derived quantities
    J = project((1.0 / resistivity_expr) * grad(V), VectorFunctionSpace(mesh, 'P', 1))  # Current density
    E = project(-grad(V), VectorFunctionSpace(mesh, 'P', 1))  # Electric field
    P = project(power_density, V_space)  # Power density
    
    # Calculate thermal gradient magnitude
    T_grad = project(grad(T), VectorFunctionSpace(mesh, 'P', 1))
    thermal_gradient = project(sqrt(dot(T_grad, T_grad)), V_space)
    
    # Calculate meaningful metrics
    max_temp = T.vector().max()
    min_temp = T.vector().min()
    avg_temp = T.vector().sum() / T.vector().size()
    
    max_delay = C.vector().max()
    avg_delay = C.vector().sum() / C.vector().size()
    
    max_gradient = thermal_gradient.vector().max()
    avg_gradient = thermal_gradient.vector().sum() / thermal_gradient.vector().size()
    
    # Clock skew is the maximum difference in delays across the chip
    clock_skew = max_delay
    
    # Compute elapsed CPU time
    cpu_time = time.time() - start_time
    
    # Return results
    return {
        'mesh': mesh,
        'voltage': V,
        'temperature': T,
        'clock_delay': C,
        'current_density': J,
        'electric_field': E,
        'power_density': P,
        'thermal_gradient': thermal_gradient,
        'convergence': errors,
        
        # Performance metrics
        'max_temperature': max_temp,
        'min_temperature': min_temp,
        'avg_temperature': avg_temp,
        'temp_range': max_temp - min_temp,
        'max_thermal_gradient': max_gradient,
        'avg_thermal_gradient': avg_gradient,
        'clock_skew': clock_skew,
        'cpu_time': cpu_time,
        'iterations': iteration,
    }

def plot_results(results, output_dir="./results"):
    """
    Generate plots of simulation results
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot temperature distribution
    plt.figure(figsize=(10, 8))
    plot(results['temperature'], title='Temperature Distribution (K)')
    plt.colorbar()
    plt.savefig(f"{output_dir}/temperature.png")
    
    # Plot voltage distribution
    plt.figure(figsize=(10, 8))
    plot(results['voltage'], title='Voltage Distribution (V)')
    plt.colorbar()
    plt.savefig(f"{output_dir}/voltage.png")
    
    # Plot clock delay distribution
    plt.figure(figsize=(10, 8))
    plot(results['clock_delay'], title='Clock Delay Distribution (s)')
    plt.colorbar()
    plt.savefig(f"{output_dir}/clock_delay.png")
    
    # Plot thermal gradient magnitude
    plt.figure(figsize=(10, 8))
    plot(results['thermal_gradient'], title='Thermal Gradient Magnitude (K/m)')
    plt.colorbar()
    plt.savefig(f"{output_dir}/thermal_gradient.png")
    
    # Plot convergence history
    plt.figure(figsize=(10, 6))
    plt.semilogy(results['convergence'])
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Convergence History')
    plt.savefig(f"{output_dir}/convergence.png")
    
    # Plot power density
    plt.figure(figsize=(10, 8))
    plot(results['power_density'], title='Power Density (W/m³)')
    plt.colorbar()
    plt.savefig(f"{output_dir}/power_density.png")
    
    print(f"Plots saved to {output_dir}/")
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(f"Temperature range: {results['min_temperature']:.2f} - {results['max_temperature']:.2f} K")
    print(f"Thermal gradient (max): {results['max_thermal_gradient']:.2f} K/m")
    print(f"Clock skew: {results['clock_skew']*1e12:.3f} ps")
    print(f"CPU time: {results['cpu_time']:.2f} seconds")


def validate_electro_thermal_timing_simulation():
    # Run the simulation
    results = electro_thermal_timing_simulation()

    # 1. Validate Temperature Range
    temp_range = results['temp_range']
    assert 0 < temp_range < 50, f"Temperature range seems abnormal: {temp_range} K"

    # 2. Validate Thermal Gradient
    max_thermal_gradient = results['max_thermal_gradient']
    assert 0 < max_thermal_gradient < 100, f"Max thermal gradient is too high: {max_thermal_gradient} K/m"

    # 3. Validate Clock Skew
    clock_skew = results['clock_skew'] * 1e12  # Convert to ps for easier interpretation
    assert 0 < clock_skew < 1000, f"Clock skew seems too high: {clock_skew} ps"

    # 4. Validate Power Density
    power_density = results['power_density']
    
    # To evaluate the power_density, you should use `power_density.vector().get_local()` to extract the values
    power_density_values = power_density.vector().get_local()
    
    # Check that all power densities are positive
    assert np.all(power_density_values > 0), "Power density contains non-positive values!"

    # 5. Validate Convergence
    convergence = results['convergence']
    assert len(convergence) > 0, "Convergence history is empty!"
    assert convergence[-1] < 1e-4, f"Convergence not sufficient: {convergence[-1]}"

    # 6. Check CPU Time
    cpu_time = results['cpu_time']
    assert cpu_time < 300, f"CPU time is too high: {cpu_time} seconds"

    # If all checks pass, print a success message
    print("Electro-Thermal Timing Simulation passed validation successfully!")

# Run the validation test
validate_electro_thermal_timing_simulation()
