import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import LogLocator, MaxNLocator
from matplotlib.path import Path

# ---------------------------------------------------------------
# TEST 1
# ---------------------------------------------------------------


def plot_grid(filename):
    with np.load(filename) as data:
        iters, times, taus, thetas = data['iters'], data['times'], data['taus'], data['thetas']
        names = data['names'].tolist()

    names[0] = r'Malitsky$\endash$Tam'
    names[2] = r'Alcantara$\endash$Dao$\endash$Takeda'

    med = np.mean(iters.mean(axis=4), axis=3)

    vmin, vmax = med.min(), med.max()
    log = vmax / vmin >= 100
    cnorm = LogNorm(vmin, vmax) if log else Normalize(vmin, vmax)
    levels = np.r_[np.arange(125, 250, 50), np.arange(300, 800, 100), np.arange(1000, 3501, 500)]

    loc = LogLocator(subs=(1, 1.5, 2, 3, 5, 7)) if log else MaxNLocator(15)
    ticks_bar = loc.tick_values(vmin, vmax)
    levels_bar = ticks_bar[(ticks_bar >= vmin) & (ticks_bar <= vmax)]

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['cmr10'],
        'mathtext.fontset': 'cm',
        'axes.formatter.use_mathtext': True,
        'font.size': 22,
    })

    fig, axes = plt.subplots(2, 3, figsize=(27 , 17), constrained_layout=True)
    for ax, name, M in zip(axes.flat, names, med):
        mesh = ax.pcolormesh(taus, thetas, M.T, cmap='viridis_r', norm=cnorm, shading='gouraud', alpha=0.7)
        ax.clabel(ax.contour(taus, thetas, M.T, levels=levels, colors='k', linewidths=0.6, alpha=0.65))
        i, j = np.unravel_index(M.argmin(), M.shape)
        ax.plot(taus[i], thetas[j], 'o', color='red', mec='white', ms=12, clip_on=False)
        ax.set(title=f'{name}\n' rf'$(\tau, \theta) = ({taus[i]:g}, {thetas[j]:g})$', xlabel=r'$\tau$', ylabel=r'$\theta$')
    color_bar = fig.colorbar(mesh, ax=axes, ticks=levels_bar, format='%g', label='Iterations', location='top', pad=0.01, aspect=60)
    color_bar.ax.invert_xaxis()
    color_bar.set_label('Iterations', fontsize=35, labelpad=20)
    fig.canvas.draw()
    plt.savefig('optimal_parameters.pdf', dpi=150)


# ---------------------------------------------------------------
# TEST 2
# ---------------------------------------------------------------

def plot_performance(filename):
    with np.load(filename) as data:
        names = [k for k in data.files if not k.endswith(('_iterations', '_tau', '_theta'))]
        times = {name: data[name] for name in names}
    labels = list(names)
    labels[0] = r'Malitsky$\endash$Tam'
    labels[2] = r'Alcantara$\endash$Dao$\endash$Takeda'
    Tmax=6
    colors = plt.cm.tab10.colors
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['cmr10'],
        'mathtext.fontset': 'cm',
        'axes.formatter.use_mathtext': True,
    })
    fig, ax = plt.subplots(figsize=(10, 5))
    min_times = np.minimum.reduce([np.array(times[alg]) for alg in names])
    r = {alg: np.array(times[alg]) / min_times for alg in names}
    P = {alg: (lambda t, a=alg: np.mean(r[a] <= t)) for alg in names}   
    T=np.linspace(1,Tmax,num=1000)
    draw_order = [4, 3, 2, 1, 5, 0]
    zorder_of = {idx: 2 + rank for rank, idx in enumerate(draw_order)}
    for idx, alg in enumerate(names):
        ax.plot(
            T,
            [P[alg](t) for t in T],
            linewidth=2,
            markersize=5,
            color=colors[idx % len(colors)],
            label=labels[idx],
            zorder=zorder_of[idx],
        )
    ax.set_xlabel(r'Factor $\alpha$', fontsize=15)
    ax.set_ylabel(r'Distribution function $\pi_S(\alpha)$', fontsize=15)
    ax.set_xticks(range(1, Tmax + 1))
    ax.set_yticks(np.linspace(0,1,11))
    ax.tick_params(axis='both', labelsize=15)
    ax.legend(title="Methods", loc="best", fontsize=12, title_fontsize=13,)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("times_admm_profile.pdf", dpi=150, bbox_inches="tight")