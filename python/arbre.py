import matplotlib.pyplot as plt
import networkx as nx

def arborize(L):
    tree = {L[0][0]:set()}
    for i in range (len(L)):
        for j in range(len(L[i])):
            if j != len(L[i])-1 :
                if L[i][j] in tree:
                    tree[L[i][j]].add(L[i][j+1])
                    # comment gerer deux chemins différents qui menent à la même feuille? faire avec les sets et ajouter à tout les coups
                else :
                    tree[L[i][j]] = {L[i][j+1]}
    return tree


def visualize_tree_as_graph(tree):
    G = nx.DiGraph(tree)
    node_colors = ["red" if node[1] == 0 else "yellow" if node[1] == 1 else "green" if node[1] == 2 else "blue" for node in G.nodes()]
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    nx.draw(G, pos, with_labels=False, node_size=50, node_color=node_colors, font_size=10, font_weight="bold")
    plt.show()

def directed_to_undirected(directed_graph):
    undirected_graph = {}  # Initialiser le graphe non-orienté
    # Parcourir chaque clé-valeur dans le graphe orienté
    for key, values in directed_graph.items():
        # Ajouter la clé dans le graphe non-orienté avec ses valeurs
        if key in undirected_graph:
            undirected_graph[key].update(values)
        else:
            undirected_graph[key] = set(values)
        # Pour chaque valeur, ajouter la clé comme une arête non-orientée
        for value in values:
            if value in undirected_graph:
                undirected_graph[value].add(key)
            else:
                undirected_graph[value] = {key}
    return undirected_graph
