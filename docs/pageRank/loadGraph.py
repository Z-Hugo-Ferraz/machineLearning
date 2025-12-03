import networkx as nx

path = "pageRankData/roadNet-CA.txt"

G = nx.read_edgelist(
    path,
    comments="#",
    nodetype=int,
    create_using=nx.DiGraph()
)

print(nx.info(G))