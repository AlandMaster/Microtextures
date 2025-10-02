import microtextures
import csv
import os
import numpy as np
import time
import PyAnsys.PyFluent_main as pyfluent_main
import PyAnsys.SC_cmd as sc_cmd
import PyAnsys.boundarylayerthickness as blt
from tomlkit import parse

# ------------------------------
# Core pipeline (unchanged)
# ------------------------------
def main(depth, width, gap):
    tick = time.time()

    with open('config.toml', 'r') as file:
        toml_string = file.read()
        config = parse(toml_string)

    fp_testbed = (config["filepaths"]["models"])

    # Change working directory
    os.chdir(fp_testbed)

    # Create model
    model = microtextures.Scallop(depth, width, gap, 0).texture_disc()
    microtextures.CQModel().export_STEP(model, 8)
    
    # Store a copy of the model with identifiers in the name
    microtextures.CQModel().export_STEP(model, r'_' + str(depth) + r'_' + str(width) + r'_' + str(gap))

    tock = time.time()
    print("Model creation time taken: ", tock - tick)

    # Execute SC script
    tick = time.time()
    sc_cmd.execute()
    tock = time.time()
    print("SC script execution time taken: ", tock - tick)
    
    # Restore working directory (SC script changes it)
    os.chdir(fp_testbed)
    
    # Run meshing and simulations in Fluent
    tick = time.time()
    pyfluent_main.main()
    tock = time.time()
    print("Fluent meshing and simulation time taken: ", tock - tick)

    # Read the resulting CSV file and calculate the boundary layer thickness
    tick = time.time()
    y_value = blt.execute()
    tock = time.time()
    print("Boundary layer thickness calculation time taken: ", tock - tick)
    
    return y_value

# ------------------------------
# Optimisation setup (same bounds/step)
# ------------------------------
lb = np.array([0.05, 0.05, 0.05], dtype=float)  # Lower bounds
ub = np.array([0.10, 0.10, 0.10], dtype=float)  # Upper bounds
step_size = 0.001                                 # Grid step

results = []  # [depth, width, gap, thickness]

def objective_function(params):
    depth, width, gap = params
    thickness = main(depth, width, gap)
    results.append([depth, width, gap, thickness])
    return thickness

# ------------------------------
# Helpers (grid snap + neighbours)
# ------------------------------
def clip_and_snap(x, lower, upper, step):
    x = np.clip(x, lower, upper)
    k = np.round((x - lower) / step)
    return lower + k * step

def random_neighbour(x, step, rng):
    # choose one or more coordinates to change by ±1 step
    mask = rng.integers(0, 2, size=3).astype(bool)
    if not mask.any():
        mask[rng.integers(0, 3)] = True
    delta = np.zeros(3, dtype=float)
    delta[mask] = rng.choice([-step, step], size=mask.sum())
    return delta

# ------------------------------
# Simulated Annealing
# ------------------------------
def simulated_annealing(objective, lower, upper, step,
                        initial_temp=1.0,
                        final_temp=1e-3,
                        cooling_rate=0.90,
                        iters_per_temp=8,
                        seed=42):
    """
    Minimisation SA on stepped 3D grid.
    - Same bounds and step_size as PSO script.
    - Metropolis acceptance with geometric cooling.
    """
    rng = np.random.default_rng(seed)

    # Start at midpoint (snapped to grid)
    x = clip_and_snap((lower + upper) / 2.0, lower, upper, step)
    f_x = objective(x)

    best_x = x.copy()
    best_f = f_x

    T = initial_temp
    iter_count = 0

    while T > final_temp:
        for _ in range(iters_per_temp):
            iter_count += 1

            # Propose neighbour (±1 step in 1–3 dimensions)
            x_prop = clip_and_snap(x + random_neighbour(x, step, rng), lower, upper, step)
            # If no move (rare due to snapping), propose again
            if np.allclose(x_prop, x):
                x_prop = clip_and_snap(x + random_neighbour(x, step, rng), lower, upper, step)

            f_prop = objective(x_prop)
            delta = f_prop - f_x

            # Accept if better; else accept with probability exp(-delta/T)
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-12)):
                x = x_prop
                f_x = f_prop

            # Track global best
            if f_x < best_f:
                best_f = f_x
                best_x = x.copy()

            print(f"[Iter {iter_count:04d}] T={T:.5f} | f={f_x:.6f} | best={best_f:.6f} @ {best_x}")

        # Geometric cooling
        T *= cooling_rate

    return best_x, best_f

# ------------------------------
# Run SA (replacing PSO)
# ------------------------------
if __name__ == "__main__":
    best_params, best_thickness = simulated_annealing(
        objective_function,
        lb, ub,
        step_size,
        initial_temp=1.0,
        final_temp=1e-3,
        cooling_rate=0.90,
        iters_per_temp=8,
        seed=42
    )

    print("Overall Best Parameters (Depth, Width, Gap):", best_params)
    print("Overall Best Boundary Layer Thickness:", best_thickness)

    # Save the results to a CSV file (same format as original)
    with open('boundary_layer_thickness_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Depth', 'Width', 'Gap', 'Boundary Layer Thickness'])
        writer.writerows(results)
