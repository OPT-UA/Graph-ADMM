import numpy as np
import graph as gp
from tests import test_grid, test_performance, load_best
from graphics import plot_grid, plot_performance

# ---------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------

# PARAMETERS
n = 6 # number of operators
m = 100 # dimension of G=R^m
seed = 2026 # random seed
n_problems = 5 # number of random problems (TEST 1)
n_starts = 5  # number of random initial points (TEST 1)
repeats = 200 # number of instances (TEST 2)

# GRAPHS
K = gp.complete(n)
R = gp.ring(n)
Pu = gp.parallel_up(n)
Pd = gp.parallel_down(n)
S = gp.sequential(n)
B = gp.biparallel(n)

# SUB-DICTIONARY 1: ALGORITHMS   (G,G')
algorithms = {
    'Malitsky-Tam':          (R, S),
    'Sequential':            (S, S),
    'Alcantara-Dao-Takeda': (Pu, Pu),
    'Parallel Down':        (Pd, Pd),
    'Complete':              (K, K),
    'Generalized Ryu':       (K, Pd),
}

# SUB-DICTIONARY 2: PARAMETERS
parameters = {
    "n_op":    n,
    "m":       m,
    "n_problems": n_problems,
    "n_starts": n_starts,
    "repeats": repeats,
    "seed": seed,
}

# DICTIONARY
info = {
    "parameters": parameters,
    "algorithms": algorithms
    }

# ---------------------------------------------------------------
# TEST 1: STEPSIZES & RELAXATION
# ---------------------------------------------------------------

parameters["tol"] = 1e-4
parameters["tau"] = np.linspace(0.4, 4, 37)
parameters["theta"] = np.linspace(1, 1.95, 20)
filename = 'grid.npz'

best_performing = test_grid(info,filename)
plot_grid(filename)

# ---------------------------------------------------------------
# TEST 2: PERFORMANCE PROFILES
# ---------------------------------------------------------------

best = load_best()

parameters["tol"] = 1e-6
parameters["tau"] = {name: values[0] for name, values in best.items()}
parameters["theta"] = {name: values[1] for name, values in best.items()}
filename = 'performance.npz'

results = test_performance(info, filename)
plot_performance(filename)
