# Graph-ADMM

Authors: **Rubén Campoy, Scott B. Lindstrom, César López-Pastor**

# Organization of the folder

Run `main.py` to obtain the figures in the paper. In particular, it generates the following files:
- `grid.npz`: contains the data produced by Experiment 1.
- `best.npz`: contains the best step sizes and relaxation parameters for each algorithm.
- `optimal_parameters.pdf`: Figure 1 in the paper.
- `performance.npz`: contains the data produced by Experiment 2.
- `times_admm_profile.pdf`: Figure 2 in the paper.

This script uses the following Python files:
- `graph.py`: gathers everything related to the algorithmic graphs and their associated matrices.
- `quadratic.py`: defines the ADMM algorithm for minimizing quadratics.
- `tests.py`: describes both experiments, which produce the `.npz` files.
- `graphics.py`: plots the figures appearing in the paper.
