import random
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import arbre
import rdc
import Known_q
import numpy as np
from scipy.spatial import Delaunay

def gen_random_vector(max_value, n):
    """
    Generate a random vector uniformly in {0,`max_value`}^`n`.

    Args:
        max_value (int): upper bound max value for vector values.
        `n` (int): dimension of the vector.

    Returns:
        tuple: a random vector of dimension `n`.
    """
    return tuple([random.randint(0, max_value) for _ in range(n)])


def distance(e, q):
    """
    Compute the euclidean distance between vectors `e` and `q`.

    Args:
        `e` (tuple): a vector.
        `q` (tuple): another vector.
    Returns:
        float: the euclidean distance between `e` and `q`.
    """
    assert len(e) == len(q)
    return math.sqrt(sum((x2 - x1) ** 2 for x1, x2 in zip(e, q)))

def extract_nearest_element(C, q):
    """
    Extract nearest element to the vector `q` from a list of vectors `C`. The
    list `C` is modified and nearest element is removed. The distance is
    computed with euclidean distance.

    Args:
        `C` (list): the list of vectors from which to look for nearest element.
        `q` (tuple): the vector we need the closest element to.

    Returns:
        tuple: the nearest element from `C` to `q`.
    """
    min_d = float('inf')
    nearest_element = None
    for element in C:
        d = distance(element, q)
        if d < min_d:
            min_d = d
            nearest_element = element
    C.remove(nearest_element)
    return nearest_element

def get_furthest_element(W, q):
    """
    Find furthest element to the vector `q` from a list of vectors `W`. The
    list `W` is not modified. The distance is computed with euclidean distance.

    Args:
        `W` (list): the list of vectors from which to look for furthest element.
        `q` (tuple): the vector we need the closest element to.

    Returns:
        tuple: the nearest element from `W` to `q`.
    """
    max_distance = -1
    furthest_element = None
    for element in W:
        d = distance(element, q)
        if d > max_distance:
            max_distance = d
            furthest_element = element
    return furthest_element

class HNSW:
    def __init__(self, n, M, Mmax, mL, efC = 10, nodes = []):
        assert M <= Mmax
        self.n = n
        self.max_layer = -1
        self.M = M
        self.Mmax = Mmax
        self.mL = mL
        self.graph = {}
        self.is_empty = True
        self.enter_point = ()
        for node in nodes:
            self.insert(node, efC)

    def check_graph(self):
        for layer in self.graph:
            subgraph = self.graph[layer]
            for node in subgraph:
                assert len(node) == self.n

    def add_layers_until(self, l, q):
        # called when `l` > `self.max_layer`
        for i in range(self.max_layer + 1, l + 1):
            self.graph[i] = {q: []}
        self.max_layer = l
        self.enter_point = q

    def neighborhood(self, c, lc):
        return self.graph[lc][c]

    def select_neighbors_simple(self, q, C, M):
        neighbors = []
        C_=  C.copy()
        for _ in range(min(M, len(C_))):
            neighbors.append(extract_nearest_element(C_, q))
        return neighbors

    def select_neighbors_heuristic(self, q, C, M, lc, extendCandidates=True, keepPrunedConnections=True):
        R = []
        W = C.copy()
        if extendCandidates:  # extend candidates by their neighbors
            for e in C:
                for eadj in self.neighborhood(e, lc):#WARN!NG
                    if eadj not in W:
                        W.append(eadj)
        Wd = []
        while len(W) > 0 and len(R) < M:
            e = extract_nearest_element(W, q)
            if self.is_closer_to_q(e, q, R):
                R.append(e)
            else:
                Wd.append(e)
        if keepPrunedConnections:  # add some of the discarded connections from Wd
            while len(Wd) > 0 and len(R) < M:
                R.append(extract_nearest_element(Wd, q))
        return R

    def is_closer_to_q(self,e,q,R):
        distance_to_point1 = distance(e, q)
        for point in R:
            if distance(q, point) < distance_to_point1:
                return False
        return True





    def bidirectional_connect(self, neighbors, q, lc):

        self.graph[lc][q] = neighbors
        neighbors_ = neighbors.copy()
        for e in neighbors_:
            self.graph[lc][e].append(q)
            upper_bound = self.Mmax
            if lc == 0:
                upper_bound *= 2

            if len(self.graph[lc][e]) > upper_bound:
                L=self.select_neighbors_heuristic(e,self.graph[lc][e],upper_bound,lc)
                for f in self.graph[lc][e]:
                    if f not in L:
                        self.graph[lc][e].remove(f)
                        self.graph[lc][f].remove(e)
                '''
                f = get_furthest_element(self.graph[lc][e], e)
                self.graph[lc][e].remove(f)
                self.graph[lc][f].remove(e)'''


        if len(self.graph[lc][q]) == 0:
            print('coucou',lc)



    def search_layer(self, q, lc, ep, ef):
        C = ep.copy()
        v = ep.copy()
        W = ep.copy()
        while C:
            c = extract_nearest_element(C, q)
            f = get_furthest_element(W, q)
            if distance(c, q) > distance(f, q):
                break
            for e in self.neighborhood(c, lc):
                if e not in v:
                    v.append(e)
                    f = get_furthest_element(W, q)
                    if distance(e, q) < distance(f, q) or len(W) < ef:
                        C.append(e)
                        W.append(e)

                        if len(W) > ef:
                            W.remove(f)
        return W


    def search_layer_leak(self, q, lc, ep, ef,Leak):

        C = ep.copy()
        v = ep.copy()
        W = ep.copy()

        while C:

            c = extract_nearest_element(C, q)
            f = get_furthest_element(W, q)

            if distance(c, q) > distance(f, q):
                break

            temp=(c , lc)
            Leak.append(temp)
            for e in self.neighborhood(c, lc):
                if e not in v:
                    v.append(e)
                    f = get_furthest_element(W, q)
                    if distance(e, q) < distance(f, q) or len(W) < ef:
                        C.append(e)
                        W.append(e)

                        if len(W) > ef:
                            W.remove(f)
        return W


    def insert(self, q, efC):
        self.check_graph()
        assert len(q) == self.n
        W = []
        ep = [self.enter_point]
        L = self.max_layer

        l = int(-math.log(random.random()) * self.mL)
        for lc in range(L, l, -1):
            W = self.search_layer(q, lc, ep, ef=1)
            min_ = float('inf')#TODOOOOOOOOO
            for e in W:
                d = distance(e, q)
                if d < min_:
                    min_ = d
                    ep = [e]

        for lc in range(min(L, l), -1, -1):
            W = self.search_layer(q, lc, ep, efC)
            neighbors = self.select_neighbors_heuristic(q, W, self.M,lc) #lc en argument pour l'heuristic
            self.bidirectional_connect(neighbors, q, lc)
            ep = W

        if l > L:
            self.add_layers_until(l, q)

    def knn_search(self, q, K, ef, Leak):
        W = []
        ep = [self.enter_point]
        L = self.max_layer
        for lc in range(L, 0, -1):
            W = self.search_layer_leak(q, lc, ep, 1, Leak)
            min_ = float('inf')
            for e in W:
                d = distance(e, q)
                if d < min_:
                    min_ = d
                    ep = [e]
        W = self.search_layer_leak(q, 0, ep, ef, Leak)
        return self.select_neighbors_simple(q, W, K), Leak


def plot_hnsw_graph(hnsw):
    # Récupérer les nœuds et les voisins du niveau zéro
    nodes = hnsw.graph.get(0, {})
    # Créer un graphe pour le niveau zéro
    G = nx.Graph()
    # Ajouter les arêtes
    for node, neighbors in nodes.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    # Récupérer les positions des sommets
    pos = {node: (node[0], node[1]) for node in G.nodes()}
    # Dessiner le graphe
    plt.figure(figsize=(5, 5))
    nx.draw(G, pos=pos, with_labels=False, node_size=50, font_size=10)
    plt.title('Level 0', fontsize=12)
    plt.show()


'''
def plot_hnsw_graph(hnsw):
    # nombre total de niveaux
    num_levels = len(hnsw.graph)
    # Définissez la taille de la figure pour afficher les sous-graphes côte à côte
    plt.figure(figsize=(5 * num_levels, 5))
    # Parcourez chaque niveau du graphe HNSW
    for i, (level, nodes) in enumerate(hnsw.graph.items(), start=1):
        # Créez un sous-graph pour ce niveau
        G = nx.Graph()
        # Ajoutez les arêtes
        for node, neighbors in nodes.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
        # Récupérer les positions des sommets
        pos = {node: (node[0], node[1]) for node in G.nodes()}
        plt.subplot(1, num_levels, i)
        if level != 0:
            nx.draw(G, pos=pos, with_labels=False, node_size=50, font_size=10)
        else :
            nx.draw(G, pos=pos, with_labels=False, node_size=50, font_size=10)
        plt.title(f'Level {level}', fontsize=12)

    plt.tight_layout()
    plt.show()'''

