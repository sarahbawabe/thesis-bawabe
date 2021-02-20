import networkx.algorithms.community as nx_comm
import parser
import kdtree
import graph as g
import networkx as nx
# from networkx.algorithms.community import greedy_modularity_communities

# G = nx.karate_club_graph()

# sorted(c[0])
# [8, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]


def find_modularity(graph, communities):
    return nx_comm.modularity(graph, communities)

def get_communities(graph):
    community_list = list(nx_comm.greedy_modularity_communities(graph))
    communities = []
    for comm in community_list:
        communities.append(nx.subgraph(graph,list(comm)))
    return communities

def main():
    snapshot = 'snap_000.0'
    ids, coords = parser.read_snapshot(snapshot)
    graph = kdtree.build_nneigh_graph(coords[:100], 3000)
    communities = get_communities(graph)
    # graph = gr.build_graph(coords[:100])
    # g.plot_3d_graph(graph)

    modularity = find_modularity(graph, communities)
    print("modularity", modularity)
    
    g.plot_3d_graphs(communities)


if __name__ == '__main__':
    main()
