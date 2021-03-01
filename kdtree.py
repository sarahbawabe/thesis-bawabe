import numpy as np
from scipy import spatial
import networkx as nx
from code.graph import *


def build_nneigh_graph(coords, radius, simple=False):
    '''
    Builds a graph where nodes are connected to all neighboring nodes within given
    radius.
    @param coords : list of coordinate locations of each node
    @param radius : size of radius determining area within which nodes connect with
                    each other (should be same units as coords)
    @return graph of nodes with given coordinates, where nodes within radius of
    each other are connected
    '''

    kdtree = spatial.KDTree(coords)
    neighs = kdtree.query_ball_point(coords, radius)
    num = 0
    for list in neighs:
        num += (len(list) - 1)
    print(num)

    graph = None
    if simple:
        graph = build_simple_graph(coords)
    else:
        graph = build_graph(coords)

    # add edges to graph
    for i in range(len(neighs)):
        if len(neighs[i]) > 1: # if has neighbors within radius
            for j in range(len(neighs[i])):
                if i != j and not graph.has_edge(j,i): # ensure no self-loops or duplicates
                    graph.add_edge(i, j)

    return graph
