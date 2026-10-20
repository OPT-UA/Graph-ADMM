import numpy as np
import graph as gp
from scipy.linalg import cho_factor, cho_solve
from time import perf_counter


def prepare_method(G, Gp, A, Q, tau, n):
    Z = gp.onto(Gp)
    degrees = np.array([G.degree(i) for i in range(n)])
    H = [degrees[i] * Q[i].T @ Q[i]+ tau * A[i].T @ A[i] for i in range(n)]
    H = [cho_factor(Hi, lower=True) for Hi in H]
    return Z, degrees, H


def initial_point(n, m, mode, rng):
    if mode == "zero": return np.zeros((n-1, m))
    if mode == "random":
        return rng.standard_normal((n-1, m))


def admm_step(w, G, Z, degrees, H, A, b_blocks, tau, theta, n, m):
    x = [None] * n
    y = np.zeros((n, m))
    r = np.zeros((n, m))
    Zw = Z @ w
    for i in range(n):
        incoming = np.zeros(m)
        for h in G.predecessors(i): incoming += y[h]
        u = Zw[i] + 2 * incoming
        rhs = A[i].T @ (tau * b_blocks[i] - u)
        x[i] = cho_solve(H[i], rhs)
        r[i] = A[i] @ x[i] - b_blocks[i]
        y[i] = (u + tau * r[i]) / degrees[i]
    w_new = w - theta * Z.T @ y
    return x, y, r, w_new


def run(G, Gp, A, Q, b_blocks, tau, theta, w0, tol, n , m):
    Z, degrees, H = prepare_method(G, Gp, A, Q, tau, n)
    w = w0.copy()
    start = perf_counter()
    error=np.inf
    k=0
    while error >= tol:
        x, y, r, w = admm_step(w, G, Z, degrees, H, A, b_blocks, tau, theta, n , m)
        error = max(np.linalg.norm(r.sum(axis=0)), np.linalg.norm(y - y.mean(axis=0)))
        k+=1
    return k, perf_counter() - start