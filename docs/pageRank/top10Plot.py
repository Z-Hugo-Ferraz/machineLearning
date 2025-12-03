import numpy as np
from io import StringIO
import networkx as nx
import matplotlib.pyplot as plt


def pagerank_custom(G, d=0.85, tol=1e-4, max_iter=100):
    nodes = list(G.nodes())
    N = len(nodes)
    idx = {n: i for i, n in enumerate(nodes)}

    pr = np.full(N, 1.0 / N)
    out_deg = np.array([G.out_degree(n) for n in nodes])
    predecessors = {i: [idx[p] for p in G.predecessors(nodes[i])] for i in range(N)}

    base = (1 - d) / N

    for _ in range(max_iter):
        pr_new = np.full(N, base)

        # Dangling nodes
        dangling_sum = pr[out_deg == 0].sum()
        pr_new += d * dangling_sum / N

        # Contribuições dos predecessores
        for i in range(N):
            s = 0
            for j in predecessors[i]:
                if out_deg[j] > 0:
                    s += pr[j] / out_deg[j]
            pr_new[i] += d * s

        # critério de parada
        if np.max(np.abs(pr_new - pr)) < tol:
            pr = pr_new
            break

        pr = pr_new

    return {nodes[i]: float(pr[i]) for i in range(N)}


def main():
    path = "pageRankData/roadNet-CA.txt"

    # Carrega grafo
    G = nx.read_edgelist(
        path,
        comments="#",
        nodetype=int,
        create_using=nx.DiGraph()
    )

    # Calcula PageRank manual
    pr_dict = pagerank_custom(G, d=0.85)

    # Top-10
    top10 = sorted(pr_dict.items(), key=lambda x: x[1], reverse=True)[:10]

    # ---------- Plot ----------
    nodes = [str(n) for n, _ in top10]
    scores = [s for _, s in top10]

    plt.figure(figsize=(10, 5))
    plt.bar(nodes, scores)
    plt.xlabel("Nós (IDs)")
    plt.ylabel("PageRank")
    plt.title("Top-10 Nós por PageRank (Implementação Manual)")
    
    buffer = StringIO()
    plt.savefig(buffer, format="svg")
    print(buffer.getvalue())
