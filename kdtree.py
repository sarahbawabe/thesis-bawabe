import numpy as np
from scipy import spatial
import networkx as nx
from code.graph import *


def build_nneigh_graph(coords, radius):
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
    graph = build_graph(coords)

    # add edges to graph
    for i in range(len(neighs)):
        if len(neighs[i]) > 1: # if has neighbors within radius
            for j in range(len(neighs[i])):
                if i != j and not graph.has_edge(i,j): # ensure no self-loops or duplicates
                    graph.add_edge(i, j)

    return graph


if __name__ == '__main__':
    obj_list = bOb.convert_to_obj_list(m, coords_matrix, vels_matrix)
    # obj_list = bOb.generate_rand_obj_list(N=10,ndim=3)
    # obj_list = [bOb.bodyObject(1,np.array([1,1,1]),np.array([1,1,1])), bOb.bodyObject(1,np.array([2,2,2]),np.array([1,1,1]))]
    nBody = NBody(obj_list, ndim=3, iters=10)
    nBody.perform_simulation()
    nneigh_graph = build_nneigh_graph(nBody.pos_matrix[:,:,0], 25)
    plot_graph(nneigh_graph)
    plot_3d_graph(nneigh_graph)
