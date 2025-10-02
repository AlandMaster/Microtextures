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

def objective_function(params, cache):
    # Snap once more (paranoia) before evaluation
    params = clip_and_snap(params, lb, ub, step_size)
    key = tuple(np.round(params, 6))
    if key in cache:
        return cache[key]
    depth, width, gap = params
    thickness = main(depth, width, gap)
    results.append([depth, width, gap, thickness])
    cache[key] = thickness
    return thickness

# ------------------------------
# Helpers (grid snap + random)
# ------------------------------
def clip_and_snap(x, lower, upper, step):
    x = np.clip(x, lower, upper)
    k = np.round((x - lower) / step)
    return lower + k * step

def random_point(lower, upper, step, rng):
    # sample uniformly on the stepped grid
    ks = rng.integers(0, np.round((upper - lower)/step).astype(int) + 1, size=3)
    return lower + ks * step

# ------------------------------
# Genetic Algorithm (real-coded on stepped grid)
# ------------------------------
def tournament_select(pop, fitness, k, rng):
    # pick k random, return index of best (minimisation)
    idxs = rng.choice(len(pop), size=k, replace=False)
    best = idxs[0]
    best_f = fitness[best]
    for i in idxs[1:]:
        if fitness[i] < best_f:
            best = i
            best_f = fitness[i]
    return best

def blx_alpha_crossover(p1, p2, alpha, rng):
    # BLX-α for each gene then snap
    cmin = np.minimum(p1, p2)
    cmax = np.maximum(p1, p2)
    I = cmax - cmin
    lower = cmin - alpha * I
    upper = cmax + alpha * I
    child = rng.uniform(lower, upper)
    return child

def mutate(x, step, mut_rate, rng):
    # With prob mut_rate per gene, move ±1 step (or stay)
    y = x.copy()
    for i in range(3):
        if rng.random() < mut_rate:
            y[i] = y[i] + rng.choice([-step, step])
    return y

def genetic_algorithm(lower, upper, step,
                      objective,
                      population_size=10,
                      generations=20,
                      crossover_rate=0.9,
                      mutation_rate=0.2,
                      tournament_k=3,
                      alpha=0.25,          # BLX-α
                      elitism=2,
                      seed=42):
    """
    Minimisation GA over a stepped 3D grid (keeps same bounds/step as PSO).
    Uses BLX-α crossover + simple ±1 step mutation, with snapping and bounds.
    """
    rng = np.random.default_rng(seed)
    cache = {}

    # Initial population: random grid points
    population = np.array([random_point(lower, upper, step, rng) for _ in range(population_size)], dtype=float)
    # Ensure uniqueness by snapping (already snapped) and possible reseeding if duplicates (optional)

    # Evaluate
    fitness = np.array([objective(ind, cache) for ind in population], dtype=float)

    # Track global best
    best_idx = int(np.argmin(fitness))
    best_x = population[best_idx].copy()
    best_f = fitness[best_idx]

    print(f"[Gen 0] best={best_f:.6f} @ {best_x}")

    for gen in range(1, generations + 1):
        # --- Elitism: carry over top elites ---
        elite_idx = np.argsort(fitness)[:elitism]
        new_pop = [population[i].copy() for i in elite_idx]

        # --- Create offspring ---
        while len(new_pop) < population_size:
            # Parent selection by tournament
            i1 = tournament_select(population, fitness, tournament_k, rng)
            i2 = tournament_select(population, fitness, tournament_k, rng)
            p1, p2 = population[i1], population[i2]

            # Crossover
            if rng.random() < crossover_rate:
                child = blx_alpha_crossover(p1, p2, alpha, rng)
            else:
                child = p1.copy()

            # Mutation
            child = mutate(child, step, mutation_rate, rng)

            # Snap to grid and clip to bounds
            child = clip_and_snap(child, lower, upper, step)
            new_pop.append(child)

        population = np.array(new_pop, dtype=float)

        # Evaluate new population
        fitness = np.array([objective(ind, cache) for ind in population], dtype=float)

        # Track global best
        gen_best_idx = int(np.argmin(fitness))
        gen_best_x = population[gen_best_idx].copy()
        gen_best_f = fitness[gen_best_idx]

        if gen_best_f < best_f:
            best_f = gen_best_f
            best_x = gen_best_x.copy()

        print(f"[Gen {gen}] best={best_f:.6f} | gen_best={gen_best_f:.6f} @ {gen_best_x}")

    return best_x, best_f

# ------------------------------
# Run GA (replacing PSO)
# ------------------------------
if __name__ == "__main__":
    best_params, best_thickness = genetic_algorithm(
        lb, ub, step_size,
        objective_function,
        population_size=10,
        generations=20,
        crossover_rate=0.9,
        mutation_rate=0.2,
        tournament_k=3,
        alpha=0.25,
        elitism=2,
        seed=42
    )

    print("Overall Best Parameters (Depth, Width, Gap):", best_params)
    print("Overall Best Boundary Layer Thickness:", best_thickness)

    # Save the results to a CSV file (same format as original)
    with open('boundary_layer_thickness_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Depth', 'Width', 'Gap', 'Boundary Layer Thickness'])
        writer.writerows(results)
