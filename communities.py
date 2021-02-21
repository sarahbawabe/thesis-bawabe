import networkx.algorithms.community as nx_comm
import parser
import kdtree
import graph as g
import networkx as nx
import numpy as np


def find_modularity(graph, communities):
    return nx_comm.modularity(graph, communities)

def get_communities(graph):
    community_list = list(nx_comm.greedy_modularity_communities(graph))
    communities = []
    for comm in community_list:
        communities.append(nx.subgraph(graph,list(comm)))
    return communities

def condense_graph(communities):
    positions = []
    for community in communities:
        node_xyz = np.array([community.nodes[v]['coords'] for v in community.nodes])
        avg_xyz = np.mean(node_xyz, axis=0)
        print(avg_xyz)
        positions.append(avg_xyz)

    return g.build_graph(np.array(positions))


def main():
    snapshot = 'snap_000.0'
    ids, coords = parser.read_snapshot(snapshot)
    graph = kdtree.build_nneigh_graph(coords[:100], 3000)
    communities = get_communities(graph)
    # graph = gr.build_graph(coords[:100])
    # g.plot_3d_graph(graph)
    modularity = find_modularity(graph, communities)
    print("modularity", modularity)

    # g.plot_3d_graphs(communities)

    condensed = condense_graph(communities)
    g.plot_3d_graph(condensed)




if __name__ == '__main__':
    main()
