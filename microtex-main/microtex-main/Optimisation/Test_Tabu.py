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
# Tabu Search (continuous with grid step)
# ------------------------------
def clip_and_snap(x, lower, upper, step):
    """Clip to [lower, upper] and snap to the step grid to respect the same discretisation."""
    x = np.clip(x, lower, upper)
    # snap to nearest grid defined by lower + k*step
    k = np.round((x - lower) / step)
    return lower + k * step

def generate_neighbourhood(x, step):
    """Generate Moore neighbourhood in 3D: all combinations of -1, 0, +1 (excluding 0,0,0)."""
    deltas = [-1, 0, 1]
    neighbourhood = []
    for dx in deltas:
        for dy in deltas:
            for dz in deltas:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                neighbourhood.append(np.array([dx, dy, dz], dtype=float) * step)
    # Shuffle for a bit of diversification
    np.random.shuffle(neighbourhood)
    return neighbourhood

def tabu_search(objective, lower, upper, step, 
                max_iter=50, 
                tabu_tenure=25, 
                patience=20, 
                seed=42):
    """
    Simple Tabu Search for 3D continuous variables on a stepped grid.
    - Keeps the same parameter bounds and step_size as the PSO script.
    - Uses aspiration criterion: tabu move allowed if it beats global best.
    """
    rng = np.random.default_rng(seed)
    # Start from the midpoint, snapped to grid
    x = clip_and_snap((lower + upper) / 2.0, lower, upper, step)
    f_x = objective(x)

    best_x = x.copy()
    best_f = f_x

    tabu_list = []            # stores tuples of parameter triples
    no_improve = 0

    for it in range(1, max_iter + 1):
        neighbourhood = generate_neighbourhood(x, step)

        candidate_best_x = None
        candidate_best_f = np.inf

        for delta in neighbourhood:
            x_n = clip_and_snap(x + delta, lower, upper, step)
            key = tuple(np.round(x_n, 6))  # stable key for tabu memory

            is_tabu = key in tabu_list

            # Aspiration: allow tabu if it would improve the global best
            if is_tabu:
                # We don't yet know its value; to avoid expensive calls,
                # quickly skip clearly duplicate points (same as current)
                if np.allclose(x_n, x):
                    continue

            # Evaluate
            f_n = objective(x_n)

            # Update neighbourhood-best (tabu allowed if beats global best later)
            if (not is_tabu and f_n < candidate_best_f) or (is_tabu and f_n < best_f and f_n < candidate_best_f):
                candidate_best_x = x_n
                candidate_best_f = f_n

            # Global best update (always allowed)
            if f_n < best_f:
                best_f = f_n
                best_x = x_n

        # If we found a candidate, move; otherwise diversify randomly
        if candidate_best_x is not None:
            x = candidate_best_x
            f_x = candidate_best_f
        else:
            # Diversification: random jump by one step in a random direction
            jump = rng.choice([-1, 1], size=3) * step
            x = clip_and_snap(x + jump, lower, upper, step)
            f_x = objective(x)

        # Update Tabu list (FIFO with length tabu_tenure)
        key = tuple(np.round(x, 6))
        tabu_list.append(key)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

        # Stopping logic based on improvement
        if np.isclose(f_x, best_f) or f_x > best_f:
            no_improve += 1
        else:
            no_improve = 0

        print(f"[Iter {it:03d}] Current f={f_x:.6f} | Best f={best_f:.6f} at x={best_x}")

        if no_improve >= patience:
            print(f"No improvement in {patience} iterations. Stopping early.")
            break

    return best_x, best_f

# ------------------------------
# Run Tabu Search (replacing PSO)
# ------------------------------
if __name__ == "__main__":
    best_params, best_thickness = tabu_search(
        objective_function,
        lb, ub,
        step_size,
        max_iter=50,       # keep modest by default; increase if budget allows
        tabu_tenure=25,    # memory size
        patience=20,       # stop early if stuck
        seed=42
    )

    print("Overall Best Parameters (Depth, Width, Gap):", best_params)
    print("Overall Best Boundary Layer Thickness:", best_thickness)

    # Save the results to a CSV file (same format as original)
    with open('boundary_layer_thickness_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Depth', 'Width', 'Gap', 'Boundary Layer Thickness'])
        writer.writerows(results)
