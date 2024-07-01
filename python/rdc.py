import numpy as np
import networkx as nx
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt


def Set_to_DLGraph(points_set,dim):
    points = np.array(list(points_set), dtype=int)
    tri = Delaunay(points)
    G = nx.Graph()
    for triangle in tri.simplices:
        for i in range(dim+1):
            for j in range(dim+1):
                if i != j :
                    G.add_edge(tuple(points[triangle[i]]), tuple(points[triangle[j]]))
                    G.add_edge(tuple(points[triangle[j]]), tuple(points[triangle[i]]))
    return G

