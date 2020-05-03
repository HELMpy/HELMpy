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
