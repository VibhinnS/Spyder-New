import streamlit as st
import matplotlib.pyplot as plt
from solver import electro_thermal_simulation
import pyvista as pv
from tempfile import TemporaryDirectory
import os
import cma
import requests

st.set_page_config(layout="wide")

stack = requests.get("http://localhost:5189/get_stack").json()

st.title("🔧 Electro-Thermal Simulation Dashboard")
st.markdown("Use the sliders to adjust design parameters, run a single simulation, or launch full optimization.")

# Sidebar sliders
st.sidebar.header("Design Parameters")
V_boundary = st.sidebar.slider("Voltage Boundary (V)", 5.0, 20.0, 10.0, step=0.5)
# h = st.sidebar.slider("Heat Transfer Coefficient h (W/m²K)", 1.0, 100.0, 10.0, step=1.0)
# TIM_thickness = st.sidebar.slider("TIM Thickness (m)", 1e-5, 0.01, 0.001, format="%.5f")

# Optimization state
if "opt_progress" not in st.session_state:
    st.session_state.opt_progress = []
    st.session_state.best_result = None

def run_single_simulation():
    results = electro_thermal_simulation(
        V_boundary=V_boundary,
        layers=stack
    )
    return results

def evaluate_design(params):
    V = params[0]
    try:
        results = electro_thermal_simulation(V_boundary=V, layers=stack)
        T = results['temperature']
        P = results['power_density']
        max_temp = T.vector().max()
        avg_temp = T.vector().sum() / T.vector().size()
        total_power = P.vector().sum()
        skew = max_temp - avg_temp
        cost = max_temp + 0.001 * total_power + 0.1 * skew
        st.session_state.opt_progress.append(cost)
        
        st.write(f"Params: V={V:.2f} → MaxT={max_temp:.2f}, "
                 f"Power={total_power:.2f}, Skew={skew:.2f}, Cost={cost:.2f}")
        
        return cost
    except Exception as e:
        st.warning(f"❌ Error with V={V:.2f}: {e}")
        return 1e6  # Penalize failures

def run_optimization():
    x0 = [10.0]  # Starting guess for Voltage
    sigma0 = 1.0
    bounds = [[5.0], [20.0]]  # Voltage range

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'bounds': bounds,
        'popsize': 4,
        'maxiter': 10
    })

    with st.spinner("Running CMA-ES optimization..."):
        while not es.stop():
            solutions = es.ask()
            costs = [evaluate_design(x) for x in solutions]
            es.tell(solutions, costs)
            es.disp()

    st.session_state.best_result = es.result

# --- Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🚀 Run Simulation"):
        results = run_single_simulation()

        T = results['temperature']
        P = results['power_density']
        max_temp = T.vector().max()
        avg_temp = T.vector().sum() / T.vector().size()
        total_power = P.vector().sum()

        st.success("Simulation Complete!")
        st.metric("Max Temperature (K)", f"{max_temp:.2f}")
        st.metric("Avg Temperature (K)", f"{avg_temp:.2f}")
        st.metric("Total Power (W)", f"{total_power:.2f}")

        with TemporaryDirectory() as tmpdir:
            vtk_path = os.path.join(tmpdir, "temp_field.vtu")
            from dolfin import File
            File(vtk_path) << T

            mesh = pv.read(vtk_path)
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(mesh, scalars="f", cmap="coolwarm")
            plotter.set_background("white")
            screenshot_path = os.path.join(tmpdir, "screenshot.png")
            plotter.screenshot(screenshot_path)

            st.image(screenshot_path, caption="Temperature Field", use_column_width=True)

        fig, ax = plt.subplots()
        ax.semilogy(results['convergence'], marker='o')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Relative Error")
        ax.set_title("Convergence History")
        ax.grid(True)
        st.pyplot(fig)

with col2:
    if st.button("🧠 Run CMA-ES Optimization"):
        st.session_state.opt_progress.clear()
        run_optimization()

    if st.session_state.best_result:
        st.success("✅ Optimization complete!")
        xbest = st.session_state.best_result.xbest
        fbest = st.session_state.best_result.fbest
        st.write(f"**Best Design**: V={xbest[0]:.2f}, h={xbest[1]:.2f}, TIM={xbest[2]:.5f}")
        st.write(f"**Minimum Cost**: {fbest:.3f}")

        fig2, ax2 = plt.subplots()
        ax2.plot(st.session_state.opt_progress, marker='o')
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Cost")
        ax2.set_title("Optimization Convergence")
        ax2.grid(True)
        st.pyplot(fig2)