import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from newNBody import *
from pylab import *
from starData import *
import bodyObject as bOb
import random


def build_graph(coords, printOut=False, fullyConnected=False):
    graph = nx.Graph()
    for i in range(len(coords)):
        # name=planet_list[i]
        # graph.add_node(i, x=x_coords[i][time], y=y_coords[i][time], z=z_coords[i][time])
        graph.add_node(i, coords=coords[i,:])
        if fullyConnected:
            for n in graph:
                if n != i:
                    nodeA = graph.nodes[n]
                    nodeB = graph.nodes[i]
                    edge_wt = math.sqrt(np.sum((nodeA['coords'] - nodeB['coords'])**2))
                    graph.add_edge(n, i, weight=edge_wt)

    # print out node list
    if printOut == True:
        for node in graph:
            print("NODE ", node, graph.nodes[node])

    return graph

def plot_graph(graph):
    nx.draw(graph, with_labels=True)
    plt.draw()
    plt.show()

def plot_3d_graph(graph, fig=None):
    ''' This code is adapted from the example located at:
    https://networkx.org/documentation/latest/auto_examples/3d_drawing/plot_basic.html
    #sphx-glr-auto-examples-3d-drawing-plot-basic-py
    '''
    # Extract node and edge positions from the stored coordinates of nodes in graph
    node_xyz = np.array([graph.nodes[v]['coords'] for v in graph.nodes])
    edge_xyz = np.array([(graph.nodes[u]['coords'], graph.nodes[v]['coords']) for u, v in graph.edges()])

    # Create the 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the nodes - alpha is scaled by "depth" automatically
    color = "#%06x" % random.randint(0, 0xFFFFFF) # give nodes random color
    ax.scatter(*node_xyz.T, s=100, color=color)

    # Plot the edges
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray")

    # Set axes labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    fig.tight_layout()

    plt.show()

def plot_3d_graphs(graph_list):
    '''
    Plots multiple graphs, each with a different random color for its nodes
    and edges.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for graph in graph_list:
        # Extract node and edge positions from the stored coordinates of nodes in graph
        node_xyz = np.array([graph.nodes[v]['coords'] for v in graph.nodes])
        edge_xyz = np.array([(graph.nodes[u]['coords'], graph.nodes[v]['coords']) for u, v in graph.edges()])

        # generate random color for nodes
        r = random.random()
        g = random.random()
        b = random.random()
        color = (r,g,b)

        # Plot the nodes - alpha is scaled by "depth" automatically
        ax.scatter(*node_xyz.T, s=100, color=color)

        # Plot the edges (& color in lighter version of node color)
        light_color = (r,g,b,0.3)
        for vizedge in edge_xyz:
            ax.plot(*vizedge.T, color=light_color)

    # Set axes labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    fig.tight_layout()

    plt.show()

def create_and_plot_mst(graph):
    tree = nx.minimum_spanning_tree(graph)
    nx.draw(tree, with_labels=True)
    plt.draw()
    plt.show()
    return tree

if __name__ == '__main__':
    # obj_list = bOb.convert_to_obj_list(m, coords_matrix, vels_matrix)
    obj_list = bOb.generate_rand_obj_list(N=10,ndim=3)
    # obj_list = [bOb.bodyObject(1,np.array([1,1,1]),np.array([1,1,1])), bOb.bodyObject(1,np.array([2,2,2]),np.array([1,1,1]))]
    nBody = NBody(obj_list, ndim=3, iters=10)
    nBody.perform_simulation()
    nBody.plot()
    # graph = build_graph(nBody.pos_matrix[:,0,:],nBody.pos_matrix[:,1,:],nBody.pos_matrix[:,2,:],9)
    graph = build_graph(nBody.pos_matrix[:,:,0], printOut=True, fullyConnected=True)
    plot_graph(graph)
    tree = create_and_plot_mst(graph)
    plot_3d_graph(tree)
