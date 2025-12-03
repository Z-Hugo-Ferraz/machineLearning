import numpy as np
import networkx as nx


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
        dangling_sum = pr[out_deg == 0].sum()
        pr_new += d * dangling_sum / N

        for i in range(N):
            s = 0.0
            for j in predecessors[i]:
                if out_deg[j] > 0:
                    s += pr[j] / out_deg[j]
            pr_new[i] += d * s

        if np.max(np.abs(pr_new - pr)) < tol:
            pr = pr_new
            break

        pr = pr_new

    pr_dict = {nodes[i]: float(pr[i]) for i in range(N)}
    return pr_dict


def main():
    path = "pageRankData/roadNet-CA.txt"

    G = nx.read_edgelist(
        path,
        comments="#",
        nodetype=int,
        create_using=nx.DiGraph()
    )

    pr_dict = pagerank_custom(G, d=0.85)

    pr_values = np.fromiter(pr_dict.values(), dtype=np.float64)

    print("\n========== RESULTADOS ==========")
    print(f"Soma dos PR: {pr_values.sum():.6f}")
    print(f"PR mínimo:  {pr_values.min():.6e}")
    print(f"PR máximo:  {pr_values.max():.6e}")

    print("\nTop-10 nós por PageRank:")
    top10 = sorted(pr_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (node, score) in enumerate(top10, 1):
        print(f"{i:2d}. nó={node} | PR={score:.6e}")


if __name__ == '__main__':
    main()
