"""
Algorithm for the analytic continuation of the power series.
"""

import numpy as np



def Epsilon(serie_completa, largo_actual):
    """
    # Epsilon Algorithm for the analytic continuation of the power series through Padé approximants

    :param serie_completa:
    :param largo_actual:
    :return:
    """
    serie = serie_completa[:largo_actual]
    continuation = np.zeros((len(serie),len(serie)+1), dtype=complex)
    continuation[0][1] = serie[0]
    for i in range(1,len(serie)):
        continuation[i][1] = continuation[i-1][1] + serie[i]
    for col in range(2,len(serie)+1):
        for row in range(0,len(serie)+1-col):
            continuation[row][col] = continuation[row+1][col-2] + 1/(continuation[row+1][col-1]-continuation[row][col-1])

    return continuation[0][len(serie)]


def Pade(serie_completa,largo_actual):
    """
    Matrix method for the analytic continuation of the power series through Padé approximants
    """
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

