import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt


# =============================
# PARAMETERS
# =============================

N_PARTICLES = 20
DIM = 3

BETA2 = 0

TARGET_NW = 1000
DT = 0.001
NSTEPS = 10000
NTHERM = int(0.3 * NSTEPS)


# =============================
# POTENTIAL
# =============================

# R: walker position matrix (Nw x N x DIM)
# beta2: potential parameter, N: number of particles
# Returns v_array: potential value for each walker (Nw,)
@njit(parallel=True, fastmath=True)
def calculate_potentials(R, beta2, N):

    nw = R.shape[0]  # number of walkers

    v_array = np.zeros(nw)  # potential of each walker

    for w in prange(nw):

        r2_sum = 0.0  # sum of squared distances from each particle to the origin
        cx = cy = cz = 0.0  # centre of mass components

        for i in range(N):  # loop over particles

            x = R[w, i, 0]  # x-coordinate of particle i in walker w
            y = R[w, i, 1]  # y-coordinate of particle i in walker w
            z = R[w, i, 2]  # z-coordinate of particle i in walker w

            r2_sum += x*x + y*y + z*z

            cx += x
            cy += y
            cz += z

        # Identity: sum of squared distances to origin
        # = sum of squared inter-particle distances / N
        # - squared magnitude of centre of mass
        rij2 = N*r2_sum - (cx*cx + cy*cy + cz*cz)

        # Potential
        v_array[w] = 0.5*(r2_sum - (beta2/N)*rij2)

    return v_array


# =============================
# BRANCHING
# =============================

@njit
def get_copy_counts_sym(V_old, V_new, ET, dt):

    # Branching weight for each walker
    w = np.exp(-0.5*dt*(V_old + V_new - 2.0*ET))

    # Number of copies of each walker
    counts = (w + np.random.random(len(w))).astype(np.int32)

    return counts


# =============================
# MAIN DMC
# =============================

def dmc_run_variable(nw_target, n_part, dim,
                     dt, beta2,
                     n_steps, n_therm):

    # Walker initialisation
    # Each walker is an (N_PARTICLES, DIM) matrix representing
    # particle positions in space.
    # R shape: (n_walkers, n_particles, dim)
    R = np.random.normal(
        0.0, 0.5,
        (nw_target, n_part, dim)
    )

    # Initial potential and reference energy
    V0 = calculate_potentials(R, beta2, n_part)
    ET = np.mean(V0)

    # Population control factor
    gamma = 0.1/dt

    # Energy history
    energies = []

    print("Analytical E =",
          1.5*(1+(n_part-1)*np.sqrt(1-beta2)))

    for step in range(n_steps):

        # Previous potential
        V_old = calculate_potentials(R, beta2, n_part)

        # Diffusion step
        sigma = np.sqrt(dt)

        R += np.random.normal(
            0.0, sigma,
            R.shape
        )

        # New potential
        V_new = calculate_potentials(R, beta2, n_part)

        # Branching
        counts = get_copy_counts_sym(
            V_old, V_new, ET, dt
        )

        # Replicate walkers according to copy counts
        R = np.repeat(R, counts, axis=0)

        # Population control
        nw_new = R.shape[0]

        if nw_new == 0:
            print("Extinction")
            break

        ET = np.mean(V_new) + gamma*np.log(
            nw_target/nw_new
        )

        # Measurement
        if step > n_therm:
            energies.append(np.mean(V_new))

        if step % 1000 == 0:
            print(f"Step {step:5d} | "
                  f"Nw={nw_new:4d} | "
                  f"ET={ET:.6f}")

    return np.array(energies)


# =============================
# EXECUTION
# =============================

if __name__ == "__main__":

    energies = dmc_run_variable(
        TARGET_NW,
        N_PARTICLES,
        DIM,
        DT,
        BETA2,
        NSTEPS,
        NTHERM
    )

    E_MC = np.mean(energies)
    E_err = np.std(energies)/np.sqrt(len(energies))

    E_exact = 1.5*(1+(N_PARTICLES-1)*np.sqrt(1-BETA2))

    print("\n" + "="*40)
    print("PURE DMC RESULTS")
    print("E_DMC   =", E_MC, "+/-", E_err)
    print("E_exact =", E_exact)
    print("Error   =",
            abs(E_MC-E_exact)/E_exact*100, "%")
    print("="*40)

    # ---- Plot: energy convergence ----
    steps_post_therm = np.arange(NTHERM + 1, NTHERM + 1 + len(energies))

    # Running mean for visual clarity
    running_mean = np.cumsum(energies) / np.arange(1, len(energies) + 1)

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(steps_post_therm, energies,
            color="steelblue", alpha=0.35, linewidth=0.8, label="DMC energy (per step)")
    ax.plot(steps_post_therm, running_mean,
            color="steelblue", linewidth=1.8, label="Running mean")
    ax.axhline(E_exact, color="tomato", linewidth=1.8,
               linestyle="--", label=f"Exact  $E_0 = {E_exact:.4f}$")
    ax.axhspan(E_MC - E_err, E_MC + E_err,
               color="steelblue", alpha=0.15, label=f"DMC mean ± s.e.  ({E_MC:.4f} ± {E_err:.4f})")

    ax.set_xlabel("DMC step")
    ax.set_ylabel("Energy (harmonic units)")
    ax.set_title(f"Pure DMC — ground-state energy convergence\n"
                 f"N={N_PARTICLES} bosons, β²={BETA2}, dt={DT}, $N_w$={TARGET_NW}")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig("pure_dmc_convergence.png", dpi=150)
    print("Plot saved -> pure_dmc_convergence.png")
    plt.show()
