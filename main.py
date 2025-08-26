from pccno import pccno
import numpy as np

# Exemplo de uso real: substitua pelos seus dados reais
# coord = np.loadtxt('coord.txt')
# adj = np.loadtxt('adj.txt', dtype=int)

# Exemplo fictício (quadrado)
coord = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
])
adj = np.array([
    [2, 4, 0, 0, 0],
    [1, 3, 0, 0, 0],
    [2, 4, 0, 0, 0],
    [1, 3, 0, 0, 0]
])

rota = pccno(coord, adj)
print("Rota Euleriana:", rota)
