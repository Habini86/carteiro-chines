from pccno import pccno, rota_carteiro_chines_geral
import numpy as np

# Exemplo de uso real: substitua pelos seus dados reais
# coord = np.loadtxt('coord.txt')
# adj = np.loadtxt('adj.txt', dtype=int)

# Exemplo baseado no grafo da imagem:
# Vértices: A, B, C, D, E, F (as letras)
# Arestas com pesos: os números na imagem representam as distâncias
# Estrutura do grafo baseada na imagem fornecida

coord = np.array([
    [0, 0],      # Vértice A
    [-2, 2],     # Vértice B 
    [2, 2],      # Vértice C
    [-2, -2],    # Vértice D
    [2, -2],     # Vértice E
    [4, 0]       # Vértice F
])

# Matriz de adjacência baseada na imagem com PESOS ESPECIFICADOS:
# A conecta com: B(peso 1), D(peso 2)
# B conecta com: A(peso 1), C(peso 3), D(peso 5)  
# C conecta com: B(peso 3), E(peso 6), F(peso 2)
# D conecta com: A(peso 2), B(peso 5), E(peso 4)
# E conecta com: C(peso 6), D(peso 4), F(peso 1)
# F conecta com: C(peso 2), E(peso 1)

# Usando matriz de pesos direta em vez de adjacência
import networkx as nx

# Criar grafo com pesos especificados
G = nx.Graph()
# Adicionar arestas com os pesos corretos
arestas_com_pesos = [
    ('A', 'B', 1),  # A-B peso 1
    ('A', 'D', 2),  # A-D peso 2
    ('B', 'C', 3),  # B-C peso 3
    ('B', 'D', 5),  # B-D peso 5
    ('C', 'E', 6),  # C-E peso 6
    ('C', 'F', 2),  # C-F peso 2
    ('D', 'E', 4),  # D-E peso 4
    ('E', 'F', 1),  # E-F peso 1
]

for v1, v2, peso in arestas_com_pesos:
    G.add_edge(v1, v2, weight=peso)

# Mapear vértices para índices para compatibilidade
mapa_vertices = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
mapa_indices = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}

# Criar matriz de distâncias baseada nos pesos especificados
n = len(mapa_vertices)
W = np.full((n, n), np.inf)
np.fill_diagonal(W, 0)

for v1, v2, peso in arestas_com_pesos:
    i, j = mapa_vertices[v1], mapa_vertices[v2]
    W[i, j] = peso
    W[j, i] = peso

print("Matriz de distâncias com pesos especificados:")
print(W)

# Função para resolver o Carteiro Chinês usando NetworkX diretamente
def carteiro_chines_networkx(G):
    """Resolve o problema do carteiro chinês usando NetworkX"""
    
    # 1. Verificar se já é Euleriano
    nos_impares = [v for v, d in G.degree() if d % 2 == 1]
    teve_repeticoes = len(nos_impares) > 0
    
    if not nos_impares:
        # Grafo já é Euleriano
        MG = nx.MultiGraph(G)
        circuito = list(nx.eulerian_circuit(MG))
        rota = [circuito[0][0]] + [edge[1] for edge in circuito]
        custo_total = sum(G[u][v]['weight'] for u, v in G.edges())
        return rota, custo_total, False
    
    # 2. Calcular menores caminhos entre todos os pares
    dist = dict(nx.all_pairs_dijkstra_path_length(G))
    
    # 3. Encontrar emparelhamento mínimo dos vértices ímpares
    import itertools
    H = nx.Graph()
    for u, v in itertools.combinations(nos_impares, 2):
        H.add_edge(u, v, weight=-dist[u][v])  # negativo para usar max_weight_matching
    matching = nx.algorithms.matching.max_weight_matching(H, maxcardinality=True)
    
    # 4. Criar multigrafo duplicando arestas nos caminhos mínimos
    MG = nx.MultiGraph(G)
    custo_repeticoes = 0
    
    for u, v in matching:
        # Encontrar caminho mínimo entre u e v
        caminho = nx.shortest_path(G, u, v, weight='weight')
        
        # Duplicar todas as arestas no caminho
        for i in range(len(caminho) - 1):
            no1, no2 = caminho[i], caminho[i + 1]
            peso = G[no1][no2]['weight']
            MG.add_edge(no1, no2, weight=peso)
            custo_repeticoes += peso
    
    # 5. Encontrar circuito Euleriano no grafo modificado
    circuito = list(nx.eulerian_circuit(MG))
    rota = [circuito[0][0]] + [edge[1] for edge in circuito]
    
    # 6. Calcular custo total
    custo_original = sum(G[u][v]['weight'] for u, v in G.edges())
    custo_total = custo_original + custo_repeticoes
    
    return rota, custo_total, teve_repeticoes

# Obter resultado completo do algoritmo
rota, custo_total, teve_repeticoes = carteiro_chines_networkx(G)

# Identificar automaticamente o tipo de rota
if teve_repeticoes:
    print("Grafo NÃO-Euleriano detectado!")
    print("Rota do Carteiro Chinês (com repetição de arestas):", rota)
    print(f"Custo total (incluindo repetições): {custo_total:.2f}")
else:
    print("Grafo Euleriano detectado!")
    print("Rota Euleriana (sem repetição de arestas):", rota)
    print(f"Custo total: {custo_total:.2f}")

print(f"Número de nós visitados: {len(rota)}")

# Mostrar o percurso detalhado
print("\n--- PERCURSO DETALHADO DO CARTEIRO ---")
print("O carteiro percorre o seguinte caminho:")

# Mapeamento de arestas para nomes
def identificar_aresta(v1, v2):
    """Identifica qual aresta está sendo percorrida"""
    # v1 e v2 já são strings ('A', 'B', etc.)
    aresta = tuple(sorted([v1, v2]))
    
    # Mapeamento das arestas com seus pesos baseado na imagem
    mapa_arestas = {
        ('A', 'B'): "A-B (peso 1)",
        ('A', 'D'): "A-D (peso 2)", 
        ('B', 'C'): "B-C (peso 3)",
        ('B', 'D'): "B-D (peso 5)",
        ('C', 'E'): "C-E (peso 6)",
        ('C', 'F'): "C-F (peso 2)",
        ('D', 'E'): "D-E (peso 4)",
        ('E', 'F'): "E-F (peso 1)"
    }
    
    return mapa_arestas.get(aresta, f"{v1}-{v2}")

# Função para encontrar em qual aresta o carteiro está
def encontrar_aresta_do_no(no, rota_pos):
    """Encontra qual aresta o carteiro estava percorrendo antes de chegar neste nó"""
    if rota_pos == 0:
        return None
    no_anterior = rota[rota_pos - 1]
    return identificar_aresta(no_anterior, no)

for i in range(len(rota)):
    if i == 0:
        print(f"1. POSIÇÃO INICIAL: Vértice {rota[i]}")
    else:
        # Identificar a aresta atual
        no_anterior = rota[i-1]
        no_atual = rota[i]
        aresta_atual = identificar_aresta(no_anterior, no_atual)
        
        # Obter peso da aresta do grafo
        peso = G[no_anterior][no_atual]['weight']
        
        print(f"{i+1}. ESTAVA: Vértice {no_anterior}")
        print(f"    PERCORREU: {aresta_atual}")
        print(f"    CHEGOU: Vértice {no_atual}")
        print(f"    PESO: {peso}")
        
        # Detectar se é repetição
        if i > 1:
            # Verificar se essa aresta já foi percorrida antes
            aresta_atual_tuple = tuple(sorted([no_anterior, no_atual]))
            arestas_percorridas = []
            for j in range(i-1):
                aresta_anterior_tuple = tuple(sorted([rota[j], rota[j+1]]))
                arestas_percorridas.append(aresta_anterior_tuple)
            
            if aresta_atual_tuple in arestas_percorridas:
                print(f"    *** ATENÇÃO: Esta aresta já foi percorrida antes! ***")

# Calcular estatísticas do percurso
custo_total_percorrido = 0
arestas_percorridas = []
arestas_repetidas = []

for i in range(len(rota) - 1):
    no1, no2 = rota[i], rota[i+1]
    peso = G[no1][no2]['weight']
    custo_total_percorrido += peso
    
    aresta = tuple(sorted([no1, no2]))
    if aresta in arestas_percorridas:
        arestas_repetidas.append(aresta)
    else:
        arestas_percorridas.append(aresta)

print(f"\n--- ESTATÍSTICAS DO PERCURSO ---")
print(f"Custo total percorrido: {custo_total_percorrido}")
print(f"Número total de passos: {len(rota)-1}")
print(f"Arestas únicas percorridas: {len(arestas_percorridas)}")
print(f"Arestas repetidas: {len(set(arestas_repetidas))}")
if arestas_repetidas:
    # Converter para nomes de arestas
    arestas_repetidas_nomes = []
    for aresta in set(arestas_repetidas):
        nome = identificar_aresta(aresta[0], aresta[1])
        arestas_repetidas_nomes.append(nome)
    print(f"Quais arestas foram repetidas: {arestas_repetidas_nomes}")

# Verificar se todas as arestas foram percorridas pelo menos uma vez
print(f"\n--- VERIFICAÇÃO DE COMPLETUDE ---")
print("Verificando se todas as arestas do grafo foram percorridas:")

# Listar todas as arestas que existem no grafo
arestas_grafo = [
    ('A', 'B'),  # peso 1
    ('A', 'D'),  # peso 2
    ('B', 'C'),  # peso 3
    ('B', 'D'),  # peso 5
    ('C', 'E'),  # peso 6
    ('C', 'F'),  # peso 2
    ('D', 'E'),  # peso 4
    ('E', 'F')   # peso 1
]

# Converter rota para arestas percorridas
arestas_percorridas_nomes = []
for i in range(len(rota) - 1):
    v1, v2 = rota[i], rota[i+1]
    aresta = tuple(sorted([v1, v2]))
    arestas_percorridas_nomes.append(aresta)

print(f"Total de arestas no grafo: {len(arestas_grafo)}")
print(f"Arestas percorridas (incluindo repetições): {len(arestas_percorridas_nomes)}")

# Verificar quais arestas não foram percorridas
arestas_nao_percorridas = []
for aresta in arestas_grafo:
    if aresta not in arestas_percorridas_nomes:
        arestas_nao_percorridas.append(aresta)

if arestas_nao_percorridas:
    print(f"❌ ERRO: As seguintes arestas NÃO foram percorridas:")
    for aresta in arestas_nao_percorridas:
        peso = None
        mapa_pesos = {
            ('A', 'B'): 1, ('A', 'D'): 2, ('B', 'C'): 3, ('B', 'D'): 5,
            ('C', 'E'): 6, ('C', 'F'): 2, ('D', 'E'): 4, ('E', 'F'): 1
        }
        peso = mapa_pesos.get(aresta, "?")
        print(f"  - {aresta[0]}-{aresta[1]} (peso {peso})")
else:
    print("✅ Todas as arestas foram percorridas pelo menos uma vez!")

print(f"\nDetalhamento das arestas percorridas:")
for i, aresta in enumerate(arestas_percorridas_nomes, 1):
    repetida = "🔄" if arestas_percorridas_nomes.count(aresta) > 1 else ""
    print(f"  {i}. {aresta[0]}-{aresta[1]} {repetida}")

# Debug: verificar o grafo criado
print(f"\n--- DEBUG DO GRAFO ---")
print(f"Vértices: {list(G.nodes())}")
print(f"Arestas com pesos:")
for u, v, data in G.edges(data=True):
    print(f"  {u}-{v}: peso {data['weight']}")

print(f"\n{'='*50}")