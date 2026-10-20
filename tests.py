import numpy as np
from quadratic import initial_point, run

# ---------------------------------------------------------------
# GENERIC COMPUTATIONS
# ---------------------------------------------------------------

def generate_instance(rng, n, m=100):
    # ell_i ~ Uniform{1,...,10}
    ell = rng.integers(1, 11, size=n).tolist()

    # p_i ~ Uniform{ell_i,...,10}
    p = [int(rng.integers(ell[i], 11)) for i in range(n)]

    # A_i in R^{m x ell_i}
    A = [rng.random((m, ell[i])) for i in range(n)]

    # Q_i in R^{p_i x ell_i}, resampled until it has full column rank
    Q = []
    for i in range(n):
        while True:
            Qi = rng.random((p[i], ell[i]))
            if np.linalg.matrix_rank(Qi) == ell[i]:
                Q.append(Qi)
                break

    # Generate a feasible point
    xhat = [rng.random(ell[i]) for i in range(n)]

    # Define b so that xhat is feasible
    b = np.sum([A[i] @ xhat[i] for i in range(n)],axis=0)
    b_blocks = np.tile(b / n, (n, 1))

    return ell, A, Q, b_blocks


def per_algorithm(value, names):
    if not isinstance(value, dict):
        value = dict(zip(names, value if np.iterable(value) else [value] * len(names)))
    return {name: float(value[name]) for name in names}


# ---------------------------------------------------------------
# TEST 1
# ---------------------------------------------------------------

def test_grid(info, filename):
    # GETTING PARAMETERS
    # Parameters
    parameters = info["parameters"]

    n = parameters["n_op"]
    m = parameters["m"]
    tol = parameters["tol"]
    seed = parameters["seed"]
    taus = parameters["tau"]
    thetas = parameters["theta"]
    n_problems = parameters["n_problems"]
    n_starts = parameters["n_starts"]

    # Graphs
    algorithms = info["algorithms"]

    # SEED
    rng = np.random.default_rng(seed)

    # Generating problems
    problems = [generate_instance(rng, n, m) for _ in range(n_problems)]
    # Generating starting points
    starts = [initial_point(n,m, "random",rng) for _ in range(n_starts)]
    # indices for each instance the algorithm is run
    shape = (len(algorithms), len(taus), len(thetas), n_problems, n_starts)
    # variables of iterations and CPU time
    iters, times = np.zeros(shape), np.zeros(shape)
    # number of instances per (tau,theta)
    total = n_problems * n_starts
    # number of digits of 'total'
    n_digits = len(str(total))

    # RUNNING ALGORITHMS

    print("\nTEST 1: STEPSIZES & RELAXATION\n")
    for s, (name, (G, Gp)) in enumerate(algorithms.items()):
        for i, tau in enumerate(taus):
            for j, theta in enumerate(thetas):
                for p, (ell, A, Q, b_blocks) in enumerate(problems):
                    for w, w0 in enumerate(starts):
                        count = p * n_starts + w + 1
                        print(f'{name}, tau: {tau:.2f}, theta: {theta:.2f} --- {count:{n_digits}}/{total}', end='\r', flush=True)
                        idx = (s, i, j, p, w)
                        iters[idx], times[idx] = run(G, Gp, A, Q, b_blocks, tau, theta, w0, tol, n, m)
        print()

    # SAVING RESULTS

    np.savez(filename, iters=iters, times=times, taus=taus, thetas=thetas, names=list(algorithms))

    # Mean over the initial points, then median over the problems: shape (algorithm, tau, theta)
    med = np.mean(iters.mean(axis=4), axis=3)
    tau, theta = [], []
    # Compute (tau,theta) minimizing iterations
    for s, name in enumerate(algorithms):
        i, j = np.unravel_index(np.argmin(med[s]), med[s].shape)
        tau.append(taus[i])
        theta.append(thetas[j])
    np.savez('best.npz', names=list(algorithms), tau=tau, theta=theta)


def load_best():
    with np.load('best.npz') as data:
        names, tau, theta = data['names'].tolist(), data['tau'].tolist(), data['theta'].tolist()
    return {name: (t, th) for name, t, th in zip(names, tau, theta)}

# ---------------------------------------------------------------
# TEST 2
# ---------------------------------------------------------------

def test_performance(info, filename):
    # GETTING PARAMETERS

    # Parameters
    parameters = info["parameters"]

    n = parameters["n_op"]
    m = parameters["m"]
    repeats = parameters["repeats"]
    tol = parameters["tol"]
    seed = parameters["seed"]

    # Algorithms
    algorithms = info["algorithms"]
    alg_name=list(algorithms)

    # Grid
    tau = per_algorithm(parameters["tau"], alg_name)
    theta = per_algorithm(parameters["theta"], alg_name)

    # SEED
    rng = np.random.default_rng(seed)

    # Times and iterations
    times = {name: [] for name in algorithms}
    iterations = {name: [] for name in algorithms}

    # RUNNING ALGORITHMS

    print("\nTEST 2: PERFORMANCE PROFILE\n")
    n_digits = len(str(repeats))

    for rep in range(repeats):

        ell, A, Q, b_blocks = generate_instance(rng=rng,n=n,m=m)
        w0 = initial_point(n, m, "zero", rng)

        print(f"Instance {rep + 1:{n_digits}}/{repeats}", end='\r')

        for name, (G, Gp) in algorithms.items():
            tau_k = tau[name]
            theta_k = theta[name]

            k, seconds = run(G, Gp, A, Q, b_blocks, tau_k, theta_k, w0, tol, n, m)

            times[name].append(seconds)
            iterations[name].append(k)
    print("\n")

    # SAVING RESULTS

    data = {}
    for name in algorithms:
        data[name] = np.asarray(times[name], dtype=float)
        data[f"{name}_iterations"] = np.asarray(iterations[name], dtype=int)
        data[f"{name}_tau"] = tau[name]
        data[f"{name}_theta"] = theta[name]
    np.savez(filename, **data)
