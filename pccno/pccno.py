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


def reconstruir_caminho(pred: np.ndarray, origem: int, destino: int) -> List[int]:
    """
    Reconstrói o caminho mínimo entre dois nós usando a matriz de predecessores.
    """
    if pred[origem, destino] == -1:
        return []
    
    caminho = []
    atual = destino
    while atual != origem:
        caminho.append(atual)
        atual = pred[origem, atual]
    caminho.append(origem)
    return caminho[::-1]


def rota_carteiro_chines_geral(coord: np.ndarray, adj: np.ndarray) -> Tuple[List[int], float, bool]:
    """
    Implementa a Rota do Carteiro Chinês Geral com Repetição de Arestas.
    
    Algoritmo:
    1. Verifica se o grafo já é Euleriano (todos os vértices têm grau par)
    2. Se não, encontra vértices de grau ímpar
    3. Calcula emparelhamento mínimo dos vértices ímpares
    4. Duplica arestas nos caminhos mínimos entre pares
    5. Encontra circuito Euleriano no grafo modificado
    
    Retorna:
    - rota: sequência de nós visitados
    - custo_total: custo total da rota (incluindo repetições)
    - teve_repeticoes: True se houve repetição de arestas
    """
    # 1. Calcular matriz de distâncias e criar grafo
    W = calcular_matriz_distancias(coord, adj)
    G = criar_grafo(W)
    
    # 2. Verificar se já é Euleriano
    nos_impares = encontrar_nos_impares(G)
    teve_repeticoes = len(nos_impares) > 0
    
    if not nos_impares:
        # Grafo já é Euleriano - encontrar circuito Euleriano direto
        MG = nx.MultiGraph(G)
        rota = [u for u, v in nx.eulerian_circuit(MG)]
        custo_total = sum(G[u][v]['weight'] for u, v in zip(rota, rota[1:] + [rota[0]]))
        return rota, custo_total, False
    
    # 3. Calcular menores caminhos entre todos os pares
    dist, pred = floyd_warshall(W)
    
    # 4. Encontrar emparelhamento mínimo dos vértices ímpares
    pares = emparelhamento_minimo(G, nos_impares, dist)
    
    # 5. Criar multigrafo duplicando arestas nos caminhos mínimos
    MG = nx.MultiGraph(G)
    custo_repeticoes = 0
    
    for u, v in pares:
        # Encontrar caminho mínimo entre u e v
        caminho = reconstruir_caminho(pred, u, v)
        
        # Duplicar todas as arestas no caminho
        for i in range(len(caminho) - 1):
            no1, no2 = caminho[i], caminho[i + 1]
            peso = W[no1, no2]
            MG.add_edge(no1, no2, weight=peso)
            custo_repeticoes += peso
    
    # 6. Encontrar circuito Euleriano no grafo modificado
    rota = [u for u, v in nx.eulerian_circuit(MG)]
    
    # 7. Calcular custo total
    custo_original = sum(G[u][v]['weight'] for u, v in G.edges())
    custo_total = custo_original + custo_repeticoes
    
    return rota, custo_total, teve_repeticoes


def pccno(coord: np.ndarray, adj: np.ndarray) -> List[int]:
    """
    Resolve o Problema do Carteiro Chinês Não-Orientado usando Rota com Repetição de Arestas.
    
    Este algoritmo sempre permite repetição de arestas quando necessário:
    1. Identifica vértices de grau ímpar
    2. Encontra emparelhamento mínimo entre eles
    3. Duplica arestas nos caminhos mínimos
    4. Encontra circuito Euleriano no grafo modificado
    
    coord: coordenadas dos nós
    adj: matriz de adjacência
    Retorna a sequência de nós da rota que permite repetições.
    """
    rota, custo_total, teve_repeticoes = rota_carteiro_chines_geral(coord, adj)
    
    if teve_repeticoes:
        print(f"Rota com repetição de arestas encontrada. Custo total: {custo_total:.2f}")
    else:
        print(f"Rota Euleriana encontrada (sem repetições). Custo total: {custo_total:.2f}")
    
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
    print("Rota com repetição de arestas:", rota)
