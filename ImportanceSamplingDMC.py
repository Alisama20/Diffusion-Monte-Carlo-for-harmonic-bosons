import numpy as np
from numba import njit, prange


# =============================
# PARAMETROS
# =============================

N_PARTICLES = 2
DIM = 3

BETA2 = 0.4
ALPHA = np.sqrt(1 - BETA2)   # casi óptimo

TARGET_NW = 2000
DT = 0.001
NSTEPS = 10000
NTHERM = int(0.3 * NSTEPS)


# =============================
# DRIFT
# =============================

@njit(parallel=True, fastmath=True)
def drift(R, alpha):

    nw, n, dim = R.shape

    F = np.empty_like(R)

    for w in prange(nw):
        for i in range(n):
            for d in range(dim):

                F[w,i,d] = -alpha * R[w,i,d]

    return F


# =============================
# ENERGÍA LOCAL
# =============================

@njit(parallel=True, fastmath=True)
def local_energy(R, alpha, beta2, n):

    nw = R.shape[0]

    EL = np.zeros(nw)

    for w in prange(nw):

        r2 = 0.0
        cx = cy = cz = 0.0

        for i in range(n):

            x = R[w,i,0]
            y = R[w,i,1]
            z = R[w,i,2]

            r2 += x*x + y*y + z*z

            cx += x
            cy += y
            cz += z

        rij2 = n*r2 - (cx*cx + cy*cy + cz*cz)

        EL[w] = (
            1.5*n*alpha
            + 0.5*(1-alpha*alpha)*r2
            - 0.5*(beta2/n)*rij2
        )

    return EL


# =============================
# Branching
# =============================


@njit
def get_copy_counts_sym(EL_old, EL_new, ET, dt):

    w = np.exp(-0.5*dt*(EL_old + EL_new - 2*ET))

    counts = (w + np.random.random(len(w))).astype(np.int32)

    return counts



# =============================
# DMC Principal (Importance Sampling)
# =============================

def dmc_run_IS(nw_target, n_part, dim, dt,
               beta2, alpha, n_steps, n_therm):

    # Inicialización compatible con Psi_T
    sigma = 1/np.sqrt(alpha)

    R = np.random.normal(
        0.0, sigma,
        (nw_target, n_part, dim)
    )

    # ET inicial
    EL0 = local_energy(R, alpha, beta2, n_part)
    ET = np.mean(EL0)

    gamma = 0.1/dt

    history = []


    print(f"Iniciando IS-DMC...  alpha={alpha:.4f}")


    for step in range(n_steps):

        # =====================
        # 1. Difusión + deriva
        # =====================

        EL_old = local_energy(R, alpha, beta2, n_part)

        F = drift(R, alpha)

        R += F*dt + np.random.normal(
            0.0, np.sqrt(dt),
            R.shape
        )


        # =====================
        # 2. Energía local
        # =====================

        EL_new = local_energy(R, alpha, beta2, n_part)


        # =====================
        # 3. Branching
        # =====================

        counts = get_copy_counts_sym(EL_old, EL_new, ET, dt)

        R = np.repeat(R, counts, axis=0)


        # =====================
        # 4. Control población
        # =====================

        nw_new = R.shape[0]

        if nw_new == 0:
            print("Extinción")
            break

        ET = np.mean(EL_new) + gamma*np.log(nw_target/nw_new)


        # =====================
        # 5. Medidas
        # =====================

        if step > n_therm:
            history.append(np.mean(EL_new))


        if step % 1000 == 0:

            print(f"Paso {step:5d} | "
                  f"Nw={nw_new:4d} | "
                  f"ET={ET:.5f}")


    return np.array(history)


# =============================
# Ejecución
# =============================

if __name__ == "__main__":

    energies = dmc_run_IS(
        TARGET_NW,
        N_PARTICLES,
        DIM,
        DT,
        BETA2,
        ALPHA,
        NSTEPS,
        NTHERM
    )


    if len(energies) > 0:

        E_MC = np.mean(energies)
        E_err = np.std(energies)/np.sqrt(len(energies))

        E_exact = 1.5*(1+(N_PARTICLES-1)*np.sqrt(1-BETA2))


        print("\n" + "="*40)

        print("RESULTADOS IS-DMC")

        print("E_DMC   =", E_MC, "+/-", E_err)

        print("E_exact =", E_exact)

        print("Error   =",
              abs(E_MC-E_exact)/E_exact*100,"%")

        print("="*40)
