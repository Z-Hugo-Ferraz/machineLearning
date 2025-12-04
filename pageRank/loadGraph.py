import networkx as nx

path = "./docs/pageRank/roadNet-CA.txt"

G = nx.read_edgelist(
    path,
    comments="#",
    nodetype=int,
    create_using=nx.DiGraph()
)

print(f"\nNúmero de nós: {G.number_of_nodes()}")
print(f"\nNúmero de arestas: {G.number_of_edges()}")
print(f"\nÉ direcionado? {G.is_directed()}")