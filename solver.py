from fenics import *
import numpy as np
import matplotlib.pyplot as plt

parameters["linear_algebra_backend"] = "PETSc"
parameters["krylov_solver"]["absolute_tolerance"] = 1e-8
parameters["krylov_solver"]["relative_tolerance"] = 1e-6
parameters["krylov_solver"]["maximum_iterations"] = 1000
parameters["form_compiler"]["cpp_optimize_flags"] = "-O2"
parameters["form_compiler"]["cache_dir"] = "clear"

def electro_thermal_simulation(
    nx=30, ny=30, nz=30,
    Lx=1.0, Ly=1.0, Lz=1.0,
    T_0=300.0,
    max_iterations=20,
    tolerance=1e-8,
    V_boundary=10.0,
    T_boundary=300.0,
    rho_0=1.0,
    alpha=0.004,
    k_0=1.0
):
    mesh = BoxMesh(Point(0, 0, 0), Point(Lx, Ly, Lz), nx, ny, nz)

    V_space = FunctionSpace(mesh, 'P', 1)
    T_space = FunctionSpace(mesh, 'P', 1)

    v = TrialFunction(V_space)
    w = TestFunction(V_space)
    t = TrialFunction(T_space)
    s = TestFunction(T_space)

    V = Function(V_space)
    T = Function(T_space)
    T_prev = Function(T_space)
    T_prev.interpolate(Constant(T_0))

    def voltage_boundary(x, on_boundary):
        return on_boundary and near(x[0], 0, 1e-14)

    def ground_boundary(x, on_boundary):
        return on_boundary and near(x[0], Lx, 1e-14)

    def temperature_boundary(x, on_boundary):
        return on_boundary

    bc_V_high = DirichletBC(V_space, Constant(V_boundary), voltage_boundary)
    bc_V_ground = DirichletBC(V_space, Constant(0.0), ground_boundary)
    bc_T = DirichletBC(T_space, Constant(T_boundary), temperature_boundary)

    bcs_V = [bc_V_high, bc_V_ground]
    bcs_T = [bc_T]

    iteration = 0
    error = 1.0
    errors = []

    print("Starting nonlinear iterations...")
    while iteration < max_iterations and error > tolerance:
        iteration += 1

        resistivity_expr = rho_0 * (1.0 + alpha * (T_prev - T_0))

        a_V = (1.0 / resistivity_expr) * dot(grad(v), grad(w)) * dx
        L_V = Constant(0.0) * w * dx
        solve(a_V == L_V, V, bcs_V)

        V_grad = grad(V)
        power_density = (1.0 / resistivity_expr) * dot(V_grad, V_grad)

        a_T = k_0 * dot(grad(t), grad(s)) * dx
        L_T = -power_density * s * dx
        solve(a_T == L_T, T, bcs_T)

        T_diff = T.copy(deepcopy=True)
        T_diff.vector().axpy(-1.0, T_prev.vector())
        error = norm(T_diff.vector()) / norm(T.vector())
        errors.append(error)

        print(f"Iteration {iteration}: error = {error:.6e}")
        T_prev.assign(T)

        if error < tolerance:
            print(f"\u2705 Converged after {iteration} iterations!")
            break

    if iteration == max_iterations and error > tolerance:
        print(f"\u26a0\ufe0f Warning: Did not converge after {max_iterations} iterations. Final error: {error:.6e}")

    J = project((1.0 / resistivity_expr) * grad(V), VectorFunctionSpace(mesh, 'P', 1))
    E = project(-grad(V), VectorFunctionSpace(mesh, 'P', 1))
    P = project(power_density, V_space)

    return {
        'mesh': mesh,
        'voltage': V,
        'temperature': T,
        'current_density': J,
        'electric_field': E,
        'power_density': P,
        'convergence': errors
    }

