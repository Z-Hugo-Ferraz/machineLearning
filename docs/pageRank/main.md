# PageRank

## 1. Introdução

O dataset *roadNet-CA* representa a malha viária da Califórnia, onde:

- **Nós** = interseções  
- **Arestas** = segmentos viários direcionados  

---

## 2. Carregamento do Grafo

=== "Resultado"

    ```python exec="on" 
        --8<-- "docs/pageRank/loadGraph.py"
    ```

=== "code"

    ```python exec="0" 
        --8<-- "docs/pageRank/loadGraph.py"
    ```

---

## 3. Implementação do PageRank Manual

A fórmula utilizada:

\[
PR(p_i) = \frac{1-d}{N} + d \sum_{p_j \in M(p_i)} \frac{PR(p_j)}{L(p_j)}
\]

```python
def pagerank_custom(G, d=0.85, tol=1e-4, max_iter=100):
    nodes = list(G.nodes())
    N = len(nodes)
    idx = {n: i for i, n in enumerate(nodes)}

    pr = np.full(N, 1 / N)
    out_deg = np.array([G.out_degree(n) for n in nodes])

    predecessors = {i: [idx[p] for p in G.predecessors(nodes[i])]
                    for i in range(N)}

    base = (1 - d) / N

    for _ in range(max_iter):
        pr_new = np.full(N, base)

        dangling_sum = pr[out_deg == 0].sum()
        pr_new += d * dangling_sum / N

        for i in range(N):
            s = 0
            for j in predecessors[i]:
                if out_deg[j] > 0:
                    s += pr[j] / out_deg[j]
            pr_new[i] += d * s

        if np.max(np.abs(pr_new - pr)) < tol:
            break

        pr = pr_new

    return {nodes[i]: float(pr[i]) for i in range(N)}
```

---

## 4. Execução do PageRank com d = 0.85

=== "Resultado"

    ```python exec="on" 
        --8<-- "docs/pageRank/page085.py"
    ```

=== "code"

    ```python exec="0" 
        --8<-- "docs/pageRank/page085.py"
    ```

---

## 5. Comparação com NetworkX

=== "Resultado"

    ```python exec="on" 
        --8<-- "docs/pageRank/pageNetworkX.py"
    ```

=== "code"

    ```python exec="0" 
        --8<-- "docs/pageRank/pageNetworkX.py"
    ```


---

## 6. Top-10 Nós Mais Importantes

=== "Resultado"

    ```python exec="on" 
        --8<-- "docs/pageRank/top10Plot.py"
    ```

=== "code"

    ```python exec="0" 
        --8<-- "docs/pageRank/top10Plot.py"
    ```

## 7. Variação do Fator de Amortecimento


=== "Resultado"

    ```python exec="on" 
        --8<-- "docs/pageRank/variacoes.py"
    ```

=== "code"

    ```python exec="0" 
        --8<-- "docs/pageRank/variacoes.py"
    ```


## 8. Análise Final


---