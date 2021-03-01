import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from code.newNBody import *
from pylab import *
from code.starData import *
import code.bodyObject as bOb
import random

BASE_NODE_COLOR = (17/255,71/255,106/255)
BASE_EDGE_COLOR = (65/255,151/255,189/255,0.1)

def build_graph(coords, printOut=False, fullyConnected=False):
    '''
    Builds a graph given the position coordinates of each node to be added. Nodes
    are knowledgable of their color, coordinates, and edge color.
    @param coords           : list of coordinate locations of each node
    @param printOut         : boolean representing whether the function should print the
                            graph to the terminal
    @param fullyConnected   : boolean representing whether every node should be
                            connected to every other node
    @return nx graph of nodes with given coordinates
    '''
    graph = nx.Graph() # basic graph
    for i in range(len(coords)):
        graph.add_node(i, coords=coords[i,:], color=BASE_NODE_COLOR, ec=BASE_EDGE_COLOR)
        if fullyConnected:
            for n in graph:
                if n != i:
                    nodeA = graph.nodes[n]
                    nodeB = graph.nodes[i]
                    edge_wt = math.sqrt(np.sum((nodeA['coords'] - nodeB['coords'])**2))
                    graph.add_edge(n, i, weight=edge_wt)

    # print out node list, if applicable
    if printOut == True:
        for node in graph:
            print("NODE ", node, graph.nodes[node])

    return graph

def build_simple_graph(coords):
    '''
    Builds a graph given the position coordinates of each node to be added.
    @param coords : list of coordinate locations of each node
    @return nx graph of nodes with given coordinates
    '''
    graph = nx.Graph() # basic graph
    for i in range(len(coords)):
        graph.add_node(i)

    return graph


def plot_graph(graph):
    '''
    Takes in an nx graph and plots it in 2-dimensions.
    '''
    nx.draw(graph, with_labels=True)
    plt.draw()
    plt.show()


def plot_3d_graph(graph):
    '''
    Takes in an nx graph with 3-dimensional position coordinates, and plots it in
    3d space. Assigns a color to the graph nodes based on the node's 'color' attribute.

    This code is adapted from the example located at:
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
    ax.scatter(*node_xyz.T, s=100, color=[graph.nodes[node]['color'] for node in graph.nodes])

    # Plot the edges
    colors = [graph.nodes[u]['ec'] for u,v in graph.edges]
    for i in range(len(edge_xyz)):
        ax.plot(*edge_xyz[i].T, color=colors[i])

    # Set axes labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    fig.tight_layout()
    plt.show()

# def plot_color_communities(graph):
#     '''
#     Relies on nodes in each community to have an associated color attribute.
#     '''



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


def get_mst(graph):
    '''
    Takes in an nx graph and finds its corresponding minimum spanning tree.
    '''
    return nx.minimum_spanning_tree(graph)


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
    tree = get_mst(graph)
    plot_3d_graph(tree)
