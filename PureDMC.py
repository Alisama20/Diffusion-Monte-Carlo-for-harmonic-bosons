import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt


# =============================
# PARAMETROS
# =============================

N_PARTICLES = 20
DIM = 3

BETA2 = 0

TARGET_NW = 1000
DT = 0.001
NSTEPS = 10000
NTHERM = int(0.3 * NSTEPS)


# =============================
# POTENCIAL
# =============================

#RECIBE R: MATRIZ DE POSICIONES DE LOS CAMINANTES (Nw x N x DIM)
#RECIBE BETA2: PARÁMETRO DEL POTENCIAL, N: NÚMERO DE PARTÍCULAS
#DEVUELVE: v_array: VECTOR DE POTENCIALES PARA CADA CAMINANTE (Nw)
@njit(parallel=True, fastmath=True)
def calculate_potentials(R, beta2, N):

    nw = R.shape[0] #MUMERO DE CAMINANTES

    v_array = np.zeros(nw) #POTENCIAL DE CADA CAMINANTE

    for w in prange(nw):

        r2_sum = 0.0 #SUMA DE LOS CUADRADOS DE LAS DISTANCIAS DE CADA PARTÍCULA AL ORIGEN
        cx = cy = cz = 0.0 #CENTRO DE MASA

        for i in range(N): #RECORRER CADA PARTÍCULA

            x = R[w,i,0] #POSICIÓN DE LA PARTÍCULA i EN LA DIMENSIÓN X DEL CAMINANTE w
            y = R[w,i,1] #POSICIÓN DE LA PARTÍCULA i EN LA DIMENSIÓN Y DEL CAMINANTE w
            z = R[w,i,2] #POSICIÓN DE LA PARTÍCULA i EN LA DIMENSIÓN Z DEL CAMINANTE w

            r2_sum += x*x + y*y + z*z 

            cx += x
            cy += y
            cz += z

        #IDENTIDAD: SUMA CUADRADOS DE LAS DISTANCIAS DE CADA PARTÍCULA AL ORIGEN 
        #= SUMA DE LOS CUADRADOS DE LAS DISTANCIAS ENTRE LAS PARTÍCULAS 
        # - SUMA DE LOS CUADRADOS DE LAS COMPONENTES DEL CENTRO DE MASA
        rij2 = N*r2_sum - (cx*cx + cy*cy + cz*cz)

        #POTENCIAL
        v_array[w] = 0.5*(r2_sum - (beta2/N)*rij2)

    return v_array


# =============================
# BRANCHING
# =============================

@njit
def get_copy_counts_sym(V_old, V_new, ET, dt):

    #PESO DE CADA CAMINANTE PARA EL BRANCHING
    w = np.exp(-0.5*dt*(V_old + V_new - 2.0*ET))

    #NÚMERO DE COPIAS DE CADA CAMINANTE
    counts = (w + np.random.random(len(w))).astype(np.int32)

    return counts


# =============================
# DMC PRINCIPAL
# =============================

def dmc_run_variable(nw_target, n_part, dim,
                     dt, beta2,
                     n_steps, n_therm):

    #INICIALIZACIÓN DE CAMINANTES
    #CADA CAMINANTE ES UNA MATRIZ DE TAMAÑO (N_PARTICLES, DIM) 
    #QUE REPRESENTA LAS POSICIONES DE LAS PARTÍCULAS EN EL ESPACIO
    #R = NUMERO DE CAMINANTES x NÚMERO DE PARTÍCULAS x DIMENSIONES
    R = np.random.normal(
        0.0, 0.5,
        (nw_target, n_part, dim)
    )


    #POTENCIAL INCIAL Y ENERGÍA DE REFERENCIA 
    V0 = calculate_potentials(R, beta2, n_part)
    ET = np.mean(V0)

    #FACTOR DE CONTROL DE POBLACIÓN
    gamma = 0.1/dt

    #ENERGÍAS
    energies = []


    print("E analítica =",
          1.5*(1+(n_part-1)*np.sqrt(1-beta2)))


    for step in range(n_steps):

        #POTENCIAL ANTERIOR
        V_old = calculate_potentials(R, beta2, n_part)

        #DIFUSIÓN
        sigma = np.sqrt(dt)

        #DERIVA
        R += np.random.normal(
            0.0, sigma,
            R.shape
        )

        #POTENCIAL NUEVO
        V_new = calculate_potentials(R, beta2, n_part)

        #BRANCHING
        counts = get_copy_counts_sym(
            V_old, V_new, ET, dt
        )

        #REPLICAR LOS CAMINANTES SEGÚN EL NÚMERO DE COPIAS CALCULADO
        R = np.repeat(R, counts, axis=0)


        #CONTROL DE POBLACIÓN
        nw_new = R.shape[0]

        if nw_new == 0:
            print("Extinción")
            break

        ET = np.mean(V_new) + gamma*np.log(
            nw_target/nw_new
        )

        #MEDICION
        if step > n_therm:
            energies.append(np.mean(V_new))


        if step % 1000 == 0:

            print(f"Paso {step:5d} | "
                  f"Nw={nw_new:4d} | "
                  f"ET={ET:.6f}")


    return np.array(energies)


# =============================
# Ejecución
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
    print("RESULTADOS DMC PURO")
    print("E_DMC   =", E_MC, "+/-", E_err)
    print("E_exact =", E_exact)
    print("Error   =",
            abs(E_MC-E_exact)/E_exact*100,"%")
    print("="*40)

