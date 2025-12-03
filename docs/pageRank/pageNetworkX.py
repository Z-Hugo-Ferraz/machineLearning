import numpy as np
import networkx as nx


def main():
    path = "pageRankData/roadNet-CA.txt"

    # Carregar grafo
    G = nx.read_edgelist(
        path,
        comments="#",
        nodetype=int,
        create_using=nx.DiGraph()
    )

    # PageRank com networkx
    pr_dict = nx.pagerank(G, alpha=0.85)

    pr_values = np.fromiter(pr_dict.values(), dtype=float)

    print("\n========== RESULTADOS (NetworkX) ==========")
    print(f"Soma dos PR: {pr_values.sum():.6f}")
    print(f"PR mínimo:  {pr_values.min():.6e}")
    print(f"PR máximo:  {pr_values.max():.6e}")

    print("\nTop-10 nós por PageRank:")
    top10 = sorted(pr_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (node, score) in enumerate(top10, 1):
        print(f"{i:2d}. nó={node} | PR={score:.6e}")


if __name__ == "__main__":
    main()
