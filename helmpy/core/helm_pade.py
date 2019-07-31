import numpy as np


def Pade(serie_completa,largo_actual):
    # Matrix method for the analytic continuation of the power series through Padé approximants
    serie = serie_completa[:largo_actual]
    L = int((len(serie)-1)/2)
    mat_c = np.zeros((L,L), dtype=complex)
    for i in range(1,L+1):
        for j in range(i,L+i):
            mat_c[i-1][j-i] = serie[j]
    vec_b = -np.linalg.solve(mat_c,serie[L+1:len(serie)])
    b = np.ones(L+1, dtype=complex)
    for i in range(1,L+1):
        b[i] = vec_b[L-i]
    a = np.zeros(L+1, dtype=complex)
    a[0] = serie[0]
    for i in range(1,L+1):
        aux = 0
        for k in range(i+1):
            aux += serie[k]*b[i-k]
        a[i] = aux
    return np.sum(a)/np.sum(b)
