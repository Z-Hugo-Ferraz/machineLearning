import numpy as np
import networkx as nx


def pagerank_custom(G, d=0.85, tol=1e-4, max_iter=100):
    # lista de nós e mapeamentos
    nodes = list(G.nodes())
    N = len(nodes)
    if N == 0:
        return {}

    idx = {n: i for i, n in enumerate(nodes)}

    # vetor inicial uniforme
    pr = np.full(N, 1.0 / N, dtype=float)

    # grau de saída de cada nó
    out_deg = np.array([G.out_degree(n) for n in nodes], dtype=float)

    # predecessores por índice (nós que apontam para i)
    predecessors = {
        i: [idx[p] for p in G.predecessors(nodes[i])]
        for i in range(N)
    }

    base = (1.0 - d) / N

    # iterações
    for _ in range(max_iter):
        pr_new = np.full(N, base, dtype=float)

        # contribuição de nós dangling (sem saída)
        dangling_sum = pr[out_deg == 0].sum()
        pr_new += d * dangling_sum / N

        # contribuição dos predecessores
        for i in range(N):
            s = 0.0
            for j in predecessors[i]:
                if out_deg[j] > 0:
                    s += pr[j] / out_deg[j]
            pr_new[i] += d * s

        # critério de parada
        if np.max(np.abs(pr_new - pr)) < tol:
            pr = pr_new
            break

        pr = pr_new

    # converte para dict {nó: score}
    pr_dict = {nodes[i]: float(pr[i]) for i in range(N)}
    return pr_dict


# ----------------------------------------------------------------------
# Script principal: roda para d = 0.5, 0.85, 0.99 e printa resultados
# ----------------------------------------------------------------------
# Caminho do arquivo roadNet-CA (ajuste se necessário)
path = "./docs/pageRank/roadNet-CA.txt"

# Carrega o grafo como DiGraph
G = nx.read_edgelist(
    path,
    comments="#",
    nodetype=int,
    create_using=nx.DiGraph()
)

# Valores de d a serem testados
d_values = [0.5, 0.85, 0.99]

for d in d_values:
    pr_dict = pagerank_custom(G, d=d, tol=1e-4, max_iter=100)

    pr_values = np.fromiter(pr_dict.values(), dtype=float)

    print(f"\n==================== RESULTADOS PARA d = {d} ====================")
    print(f"\nSoma dos PR: {pr_values.sum():.6f}")
    print(f"\nPR mínimo:  {pr_values.min():.6e}")
    print(f"\nPR máximo:  {pr_values.max():.6e}")

    # Top-10 nós
    top10 = sorted(pr_dict.items(), key=lambda x: x[1], reverse=True)[:10]

    print("\nTop-10 nós por PageRank:")
    for i, (node, score) in enumerate(top10, 1):
        print(f"\n{i:2d}. nó={node} | PR={score:.6e}")
