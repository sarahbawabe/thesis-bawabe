import networkx.algorithms.community as nx_comm
import networkx.algorithms.distance_measures as nx_dist
import code.parser as parser
import code.kdtree as kdtree
import code.graph as g
import networkx as nx
import numpy as np
import random


def get_modularity(graph, communities):
    '''
    Calculates the modularity of a given graph.
    @return number representing modularity of graph
    '''
    return nx_comm.modularity(graph, communities)


def get_communities(graph, simple=True):
    '''
    Finds communities in a given graph, such that modularity is optimized. Each
    node in each community given an updated color attribute, so that it "knows"
    which community it's a part of when being graphed.

    The list of subgraphs is important for condensing the graph based on
    communities, so that average coordinate locations can be calculated.

    @return list containing subgraphs of type nx.graph
    '''
    community_list = list(nx_comm.greedy_modularity_communities(graph))
    if not simple:
        for community in community_list:
            color, e_color = get_random_colors() # generate random color for each community
            for node in community:
                graph.nodes[node]['color'] = color
                graph.nodes[node]['ec'] = e_color

    communities = []
    for comm in community_list:
        communities.append(nx.subgraph(graph,list(comm)))

    return communities


def get_random_colors():
    '''
    Generates a random rgb color, and returns a tuple of the color and a lighter
    version of that rgb color (opacity 0.2).
    '''
    r = random.random()
    g = random.random()
    b = random.random()
    color = (r,g,b)
    e_color = (r,g,b,0.2)

    return [color, e_color]


def condense_graph(communities, radius, simple=True):
    '''
    Takes in a list of subgraphs and averages their position data to gain an
    average position coordinate. Then creates a new node with this average as its
    position, and adds to a list.
    @return graph containing condensed community nodes
    '''
    positions = []
    colors = []
    e_colors = []
    for community in communities:
        node_xyz = np.array([community.nodes[v]['coords'] for v in community.nodes])
        avg_xyz = np.mean(node_xyz, axis=0)
        positions.append(avg_xyz)
        if not simple:
            colors.append(community.nodes.get(next(iter(community.nodes.keys())))['color'])
            e_colors.append(community.nodes.get(next(iter(community.nodes.keys())))['ec'])

    graph = kdtree.build_nneigh_graph(np.array(positions), radius)

    for node in graph.nodes:
        graph.nodes[node]['coords'] = ''

    if not simple:
        for i in range(len(graph.nodes)):
            graph.nodes[i]['color'] = colors[i]
            graph.nodes[i]['ec'] = e_colors[i]

    return graph


def get_diameter(graph):
    '''
    Calculates the diameter (maximum eccentricity) of nodes of a given graph.
    @return number representing diameter of graph
    '''
    return nx_dist.diameter(graph)


def get_eccentricity(graph):
    '''
    Calculates the eccentricity of nodes of a given graph.
    @return dictionary keying each node to its eccentricity
    '''
    return nx_dist.eccentricity(graph)


def get_ecc_distribution(ecc_dict):
    '''
    Creates a dictionary totaling how many nodes had eccentricity of 1, 2, etc.
    @return dictionary keying each eccentricity to the total number of nodes with
    that eccentricity
    '''
    dist_dict = {}
    for key in ecc_dict.keys():
        if ecc_dict[key] in dist_dict:
            dist_dict[ecc_dict[key]] += 1
        else:
            dist_dict[ecc_dict[key]] = 1
    return dist_dict


def test_isomorphism(graph):
    num_nodes = nx.number_of_nodes(graph)
    num_edges = max(get_avg_connectivity(graph), 1)
    print("avg connectivity", num_edges)

    ba_graph = nx.generators.random_graphs.barabasi_albert_graph(num_nodes, num_edges)
    ba_iso = nx.algorithms.isomorphism.faster_could_be_isomorphic(graph, ba_graph)
    print("Isomorphic with barabasi_albert_graph?", ba_iso)


def get_avg_connectivity(graph):
    '''
    Finds the average connectivity (avg # of edges each node is adjacent to) of
    the graph.
    '''
    return int(np.mean(np.array(list(nx.algorithms.assortativity.average_neighbor_degree(graph).values()))))


def graph_properties(graph, communities):
    # calculate modularity of graph
    modularity = get_modularity(graph, communities)
    print("modularity", modularity)

    # calculate diameter (max eccentricity) of graph
    diameter = get_diameter(graph)
    print("diameter", diameter)

    # calculate eccentricity and eccentricity distribution of graph
    eccentricity = get_eccentricity(graph)
    ecc_dist = get_ecc_distribution(eccentricity)
    print("ecc_dist", ecc_dist)


def main():
    '''
    Function called by mainline.
    '''
    snapshot = 'snap_000.0'
    radius = 2500
    # best combos: 100 nodes & r=5000, 10 nodes & r=2500

    ids, coords = parser.read_snapshot(snapshot)
    graph = kdtree.build_nneigh_graph(coords[:10], radius)

    # test isomorphism of graph against random graph
    test_isomorphism(graph)

    # find mst of graph
    tree = g.get_mst(graph)

    # find communities and update nodes' color attributes, then plot
    communities = get_communities(graph)

    g.plot_3d_graph(graph)

    # plot graph showing condensed communities
    condensed = condense_graph(communities, radius)
    g.plot_3d_graph(condensed)


if __name__ == '__main__':
    main()
