"""
generate_plots.py
Generates Figures 1 and 2 from the report:
  E0 vs beta^2 for Pure DMC and IS-DMC across several particle numbers.

Output
------
  pure_dmc_results.png   --  2x2 grid, N = 2, 5, 10, 20
  is_dmc_results.png     --  2x2 grid, N = 10, 20, 50, 100
"""

import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt

# =============================
# PARAMETERS
# =============================

BETA2_VALUES = np.linspace(0.0, 0.95, 12)

DT      = 0.001
NSTEPS  = 1500
NTHERM  = 500
NW      = 300         # walker population
DIM     = 3


# =============================
# SHARED: POTENTIAL
# =============================

@njit(parallel=True, fastmath=True)
def calculate_potentials(R, beta2, N):
    nw = R.shape[0]
    v  = np.zeros(nw)
    for w in prange(nw):
        r2 = 0.0
        cx = cy = cz = 0.0
        for i in range(N):
            x = R[w, i, 0]; y = R[w, i, 1]; z = R[w, i, 2]
            r2 += x*x + y*y + z*z
            cx += x; cy += y; cz += z
        rij2 = N*r2 - (cx*cx + cy*cy + cz*cz)
        v[w] = 0.5*(r2 - (beta2/N)*rij2)
    return v


# =============================
# SHARED: BRANCHING
# =============================

@njit
def branching(w_old, w_new, ET, dt):
    w = np.exp(-0.5*dt*(w_old + w_new - 2.0*ET))
    return (w + np.random.random(len(w))).astype(np.int32)


# =============================
# PURE DMC
# =============================

def run_pure_dmc(N, beta2, nw_target=NW, dt=DT, nsteps=NSTEPS, ntherm=NTHERM):
    R  = np.random.normal(0.0, 0.5, (nw_target, N, DIM))
    ET = np.mean(calculate_potentials(R, beta2, N))
    gamma = 0.1 / dt
    energies = []

    for step in range(nsteps):
        V_old = calculate_potentials(R, beta2, N)
        R    += np.random.normal(0.0, np.sqrt(dt), R.shape)
        V_new = calculate_potentials(R, beta2, N)
        counts = branching(V_old, V_new, ET, dt)
        R = np.repeat(R, counts, axis=0)
        nw = R.shape[0]
        if nw == 0:
            break
        ET = np.mean(V_new) + gamma * np.log(nw_target / nw)
        if step >= ntherm:
            energies.append(np.mean(V_new))

    return np.mean(energies) if energies else np.nan


# =============================
# IS-DMC: DRIFT + LOCAL ENERGY
# =============================

@njit(parallel=True, fastmath=True)
def drift(R, alpha):
    F = np.empty_like(R)
    nw, n, dim = R.shape
    for w in prange(nw):
        for i in range(n):
            for d in range(dim):
                F[w, i, d] = -alpha * R[w, i, d]
    return F


@njit(parallel=True, fastmath=True)
def local_energy(R, alpha, beta2, n):
    nw = R.shape[0]
    EL = np.zeros(nw)
    for w in prange(nw):
        r2 = 0.0
        cx = cy = cz = 0.0
        for i in range(n):
            x = R[w, i, 0]; y = R[w, i, 1]; z = R[w, i, 2]
            r2 += x*x + y*y + z*z
            cx += x; cy += y; cz += z
        rij2 = n*r2 - (cx*cx + cy*cy + cz*cz)
        EL[w] = (1.5*n*alpha
                 + 0.5*(1 - alpha*alpha)*r2
                 - 0.5*(beta2/n)*rij2)
    return EL


def run_is_dmc(N, beta2, nw_target=NW, dt=DT, nsteps=NSTEPS, ntherm=NTHERM):
    alpha = np.sqrt(max(1 - beta2, 1e-6))
    sigma = 1.0 / np.sqrt(alpha)
    R  = np.random.normal(0.0, sigma, (nw_target, N, DIM))
    ET = np.mean(local_energy(R, alpha, beta2, N))
    gamma = 0.1 / dt
    energies = []

    for step in range(nsteps):
        EL_old = local_energy(R, alpha, beta2, N)
        R += drift(R, alpha)*dt + np.random.normal(0.0, np.sqrt(dt), R.shape)
        EL_new = local_energy(R, alpha, beta2, N)
        counts = branching(EL_old, EL_new, ET, dt)
        R = np.repeat(R, counts, axis=0)
        nw = R.shape[0]
        if nw == 0:
            break
        ET = np.mean(EL_new) + gamma * np.log(nw_target / nw)
        if step >= ntherm:
            energies.append(np.mean(EL_new))

    return np.mean(energies) if energies else np.nan


# =============================
# EXACT ENERGY
# =============================

def E_exact(N, beta2_arr):
    return 1.5 * (1 + (N - 1) * np.sqrt(1 - beta2_arr))


# =============================
# FIGURE 1 — PURE DMC
# =============================

def make_pure_dmc_figure():
    N_list = [2, 5, 10, 20]
    labels = ["(a)", "(b)", "(c)", "(d)"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, N, lbl in zip(axes, N_list, labels):
        print(f"  Pure DMC  N={N} ...")
        E_dmc = np.array([run_pure_dmc(N, b2) for b2 in BETA2_VALUES])
        E_an  = E_exact(N, BETA2_VALUES)

        ax.plot(BETA2_VALUES, E_dmc, "o-", color="steelblue",
                markersize=4, linewidth=1.4, label="DMC (numerical)")
        ax.plot(BETA2_VALUES, E_an,  "s--", color="darkorange",
                markersize=4, linewidth=1.4, label="Analytical")

        ax.set_title(f"$E_0$ vs $\\beta^2$ — Pure DMC,  $N={N}$", fontsize=10)
        ax.set_xlabel(r"$\beta^2$")
        ax.set_ylabel("Energy (a.u.)")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.text(0.02, 0.05, lbl, transform=ax.transAxes, fontsize=11, fontweight="bold")

    fig.suptitle("Ground-state energy $E_0$ vs interaction strength $\\beta^2$\n"
                 "Pure Diffusion Monte Carlo", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig("figures/pure_dmc_results.png", dpi=150, bbox_inches="tight")
    print("Saved -> pure_dmc_results.png")


# =============================
# FIGURE 2 — IS-DMC
# =============================

def make_is_dmc_figure():
    N_list = [10, 20, 50, 100]
    labels = ["(a)", "(b)", "(c)", "(d)"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, N, lbl in zip(axes, N_list, labels):
        print(f"  IS-DMC  N={N} ...")
        E_is = np.array([run_is_dmc(N, b2) for b2 in BETA2_VALUES])
        E_an = E_exact(N, BETA2_VALUES)

        ax.plot(BETA2_VALUES, E_an, "-",  color="black",
                linewidth=1.6, label="Analytical")
        ax.plot(BETA2_VALUES, E_is, "o-", color="tomato",
                markersize=4, linewidth=1.4, label="IS-DMC (numerical)")

        ax.set_title(f"$E_0$ vs $\\beta^2$ — IS-DMC,  $N={N}$", fontsize=10)
        ax.set_xlabel(r"$\beta^2$")
        ax.set_ylabel("Energy (a.u.)")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.text(0.02, 0.05, lbl, transform=ax.transAxes, fontsize=11, fontweight="bold")

    fig.suptitle("Ground-state energy $E_0$ vs interaction strength $\\beta^2$\n"
                 "Importance-Sampling Diffusion Monte Carlo", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig("figures/is_dmc_results.png", dpi=150, bbox_inches="tight")
    print("Saved -> is_dmc_results.png")


# =============================
# MAIN
# =============================

if __name__ == "__main__":
    print("=== Generating Pure DMC figure ===")
    make_pure_dmc_figure()

    print("\n=== Generating IS-DMC figure ===")
    make_is_dmc_figure()

    print("\nDone.")
