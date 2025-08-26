import numpy as np
import networkx as nx
from typing import List, Tuple


def calcular_matriz_distancias(coord: np.ndarray, adj: np.ndarray) -> np.ndarray:
    """
    Calcula a matriz de distâncias W para um grafo não orientado.
    coord: matriz de coordenadas dos nós (n, 2)
    adj: matriz de adjacência (índices dos vizinhos, 0 se não existe)
    """
    n = coord.shape[0]
    W = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(adj.shape[1]):
            viz = int(adj[i, j])
            if viz > 0:
                dist = np.linalg.norm(coord[i] - coord[viz-1])
                W[i, viz-1] = dist
                W[viz-1, i] = dist
    np.fill_diagonal(W, 0)
    return W


def floyd_warshall(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Algoritmo de Floyd-Warshall para menores caminhos e predecessores.
    Retorna matriz de distâncias mínimas e predecessores.
    """
    n = W.shape[0]
    dist = W.copy()
    pred = np.tile(np.arange(n), (n, 1)).T
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
                    pred[i, j] = pred[k, j]
    return dist, pred


def encontrar_nos_impares(G: nx.Graph) -> List[int]:
    """
    Retorna lista dos nós de grau ímpar.
    """
    return [v for v, d in G.degree() if d % 2 == 1]


def emparelhamento_minimo(G: nx.Graph, nos_impares: List[int], dist: np.ndarray) -> List[Tuple[int, int]]:
    """
    Encontra o emparelhamento mínimo dos nós ímpares usando NetworkX.
    """
    import itertools
    H = nx.Graph()
    for u, v in itertools.combinations(nos_impares, 2):
        H.add_edge(u, v, weight=-dist[u, v])  # negativo para usar max_weight_matching
    matching = nx.algorithms.matching.max_weight_matching(H, maxcardinality=True)
    return list(matching)


def criar_grafo(W: np.ndarray) -> nx.Graph:
    """
    Cria um grafo não orientado a partir da matriz de distâncias.
    """
    G = nx.Graph()
    n = W.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if np.isfinite(W[i, j]) and W[i, j] > 0:
                G.add_edge(i, j, weight=W[i, j])
    return G


def encontrar_rota_euleriana(G: nx.MultiGraph) -> List[int]:
    """
    Retorna a rota euleriana do grafo multigrafo.
    """
    return [u for u, v in nx.eulerian_circuit(G)] + [next(nx.eulerian_circuit(G))[0]]


def pccno(coord: np.ndarray, adj: np.ndarray) -> List[int]:
    """
    Resolve o Problema do Carteiro Chinês Não-Orientado.
    coord: coordenadas dos nós
    adj: matriz de adjacência
    Retorna a sequência de nós da rota euleriana.
    """
    W = calcular_matriz_distancias(coord, adj)
    G = criar_grafo(W)
    nos_impares = encontrar_nos_impares(G)
    if nos_impares:
        dist, pred = floyd_warshall(W)
        pares = emparelhamento_minimo(G, nos_impares, dist)
        MG = nx.MultiGraph(G)
        for u, v in pares:
            # Adiciona aresta duplicada para cada par
            caminho = []
            a, b = u, v
            while b != a:
                caminho.append(b)
                b = pred[a, b]
            caminho = [a] + caminho[::-1]
            for i in range(len(caminho)-1):
                MG.add_edge(caminho[i], caminho[i+1], weight=W[caminho[i], caminho[i+1]])
    else:
        MG = nx.MultiGraph(G)
    rota = [u for u, v in nx.eulerian_circuit(MG)]
    return rota

# Exemplo de uso (dados fictícios):
if __name__ == "__main__":
    # Exemplo: 4 nós em quadrado
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
