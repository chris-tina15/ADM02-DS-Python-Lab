
import networkx as nx
import matplotlib.pyplot as plt

print("Network Visualization: Undirected and Directed Graphs")
print("=" * 60)

# PART A: UNDIRECTED RESEARCH COLLABORATION NETWORK

G = nx.Graph()

collaborations = [
    ("Asha", "Bharath"),
    ("Asha", "Charu"),
    ("Bharath", "Charu"),
    ("Charu", "Deva"),
    ("Deva", "Eswar"),
    ("Deva", "Fiona"),
    ("Eswar", "Fiona"),
    ("Fiona", "Gautam"),
    ("Gautam", "Hema"),
    ("Hema", "Irfan"),
    ("Gautam", "Irfan"),
    ("Charu", "Fiona")
]

G.add_edges_from(collaborations)

print("\nPART A: Research Collaboration Network")
print("-" * 45)

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
print("Network density:", round(nx.density(G), 3))
print("Is connected:", nx.is_connected(G))

degree = dict(G.degree())
betweenness = nx.betweenness_centrality(G)

print("\nNode Centrality Analysis:")
print(f"{'Researcher':<12} {'Degree':<10} {'Betweenness'}")

for node in sorted(G.nodes(), key=lambda n: (-betweenness[n], n)):
    print(f"{node:<12} {degree[node]:<10} {betweenness[node]:.3f}")

path = nx.shortest_path(G, "Asha", "Irfan")

print("\nShortest path from Asha to Irfan:")
print(" - ".join(path))

communities = nx.algorithms.community.greedy_modularity_communities(G)

print("\nDetected Communities:")
for i, community in enumerate(communities, 1):
    print(f"Community {i}: {sorted(community)}")


# PART B: DIRECTED FLIGHT ROUTE NETWORK

D = nx.DiGraph()

routes = [
    ("Chennai", "Delhi", 1760),
    ("Delhi", "Chennai", 1760),
    ("Chennai", "Mumbai", 1030),
    ("Mumbai", "Delhi", 1150),
    ("Delhi", "Kolkata", 1300),
    ("Kolkata", "Guwahati", 520),
    ("Mumbai", "Bengaluru", 840),
    ("Bengaluru", "Chennai", 290),
    ("Bengaluru", "Kolkata", 1560),
    ("Hyderabad", "Bengaluru", 500),
    ("Chennai", "Hyderabad", 520),
    ("Kolkata", "Delhi", 1300)
]

D.add_weighted_edges_from(routes, weight="km")

print("\n\nPART B: Directed Flight Route Network")
print("-" * 45)

print("Number of airports:", D.number_of_nodes())
print("Number of routes:", D.number_of_edges())

pagerank = nx.pagerank(D)

print("\nAirport PageRank Analysis:")
print(f"{'Airport':<15} {'In-degree':<12} {'Out-degree':<12} {'PageRank'}")

for airport in sorted(D.nodes(), key=lambda n: -pagerank[n]):
    print(
        f"{airport:<15} "
        f"{D.in_degree(airport):<12} "
        f"{D.out_degree(airport):<12} "
        f"{pagerank[airport]:.3f}"
    )

shortest_route = nx.dijkstra_path(
    D, "Hyderabad", "Guwahati", weight="km"
)

distance = nx.dijkstra_path_length(
    D, "Hyderabad", "Guwahati", weight="km"
)

print("\nShortest route from Hyderabad to Guwahati:")
print(" -> ".join(shortest_route))
print("Total distance:", distance, "km")

print(
    "Can Guwahati reach any other city:",
    D.out_degree("Guwahati") > 0
)

print("Is the network strongly connected:", nx.is_strongly_connected(D))


# PART C: NETWORK VISUALIZATION

fig, ax = plt.subplots(1, 2, figsize=(18, 8))

# Collaboration network
pos1 = nx.spring_layout(G, seed=42)

community_colors = {}
for i, community in enumerate(communities):
    for node in community:
        community_colors[node] = i

node_colors = [community_colors[node] for node in G.nodes()]
node_sizes = [1000 + betweenness[node] * 3000 for node in G.nodes()]

nx.draw_networkx(
    G,
    pos1,
    ax=ax[0],
    with_labels=True,
    node_color=node_colors,
    node_size=node_sizes,
    font_size=9,
    font_weight="bold",
    edge_color="gray"
)

ax[0].set_title("Research Collaboration Network")
ax[0].axis("off")

# Flight route network
pos2 = nx.spring_layout(D, seed=42)

nx.draw_networkx_nodes(
    D, pos2, ax=ax[1],
    node_size=[1000 + pagerank[n] * 5000 for n in D.nodes()]
)

nx.draw_networkx_labels(
    D, pos2, ax=ax[1], font_size=8
)

nx.draw_networkx_edges(
    D, pos2, ax=ax[1],
    edge_color="gray",
    arrows=True,
    arrowsize=18,
    connectionstyle="arc3,rad=0.1"
)

edge_labels = nx.get_edge_attributes(D, "km")

nx.draw_networkx_edge_labels(
    D, pos2, edge_labels=edge_labels, ax=ax[1], font_size=7
)

# Highlight the shortest route
shortest_edges = list(zip(shortest_route[:-1], shortest_route[1:]))

nx.draw_networkx_edges(
    D, pos2, ax=ax[1],
    edgelist=shortest_edges,
    edge_color="red",
    width=2.5,
    arrows=True,
    arrowsize=20
)

ax[1].set_title("Directed Flight Route Network")
ax[1].axis("off")

plt.tight_layout()
plt.savefig("Experiment_20/exp20_graphs.png", dpi=150)
plt.show()

print("\nSaved figure: Experiment_20/exp20_graphs.png")
