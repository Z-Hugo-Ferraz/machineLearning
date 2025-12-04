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

A comparação entre os resultados obtidos pela implementação manual do algoritmo PageRank e aqueles gerados pela função pagerank do NetworkX demonstra que não houve diferenças significativas entre as duas abordagens. Tanto a soma total dos valores de PageRank quanto os valores mínimo e máximo coincidem exatamente nas duas execuções, indicando que o cálculo converge para a mesma distribuição de importância dos nós. Além disso, o Top-10 nós mais bem ranqueados é idêntico em ambas as metodologias, com os mesmos valores numéricos de PageRank e na mesma ordem, o que confirma a correta implementação da fórmula iterativa utilizada. Esses resultados validam plenamente a implementação manual, mostrando que ela está alinhada com o comportamento esperado de uma biblioteca consolidada como o NetworkX.

---

## 6. Top-10 Nós Mais Importantes


![Top 10 nós por PageRank](./docs/pageRank/graficoPageRank.png)


## 7. Variação do Fator de Amortecimento


=== "Resultado"

    ```python exec="on" 
    --8<-- "docs/pageRank/variacoes.py"
    ```

=== "code"

    ```python exec="0" 
    --8<-- "docs/pageRank/variacoes.py"
    ```

A análise das três variações do fator de amortecimento (d = 0.5, 0.85 e 0.99) mostra que, embora os valores de PageRank sofram pequenas alterações numéricas conforme o peso dado ao “teleporte” aumenta ou diminui, o ranking dos nós permanece essencialmente estável. Em todas as configurações, o nó 225438 ocupa a primeira posição, seguido pelos nós 287362, 241926, 748883 e 1682326, indicando que esses vértices são estruturalmente centrais na malha viária independentemente do comportamento do caminhante aleatório. Observa-se que, para valores menores de d (como 0.5), o PageRank tende a se distribuir de forma mais uniforme, reduzindo a diferença entre os nós de maior e menor importância. Já para valores mais altos (como 0.99), a influência da topologia se intensifica, ampliando levemente as diferenças entre os nós de maior conectividade. Apesar disso, o conjunto de nós mais influentes não se altera, o que demonstra que a rede possui uma estrutura robusta de hubs e que o PageRank é bastante estável em relação à escolha do fator de amortecimento neste grafo.

## 8. Análise Final

Em síntese, os experimentos realizados demonstram que a implementação manual do PageRank é plenamente consistente com a solução consolidada do NetworkX, validando a corretude do algoritmo desenvolvido. Além disso, a análise das diferentes configurações do fator de amortecimento evidencia que, embora os valores absolutos de PageRank variem conforme a probabilidade de teleporte, o conjunto de nós mais influentes permanece estável em todas as execuções, revelando a presença de estruturas centrais bem definidas na malha viária do roadNet-CA. Isso reforça a robustez do PageRank como métrica de importância em redes reais e mostra que a conectividade intrínseca do grafo exerce maior impacto no ranking do que ajustes finos no parâmetro d. Dessa forma, os resultados obtidos fornecem uma caracterização consistente dos principais hubs da rede e confirmam a confiabilidade tanto da abordagem manual quanto das ferramentas de análise oferecidas por bibliotecas especializadas.

---