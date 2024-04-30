from hnsw import *
import Known_q
import rdc
import arbre
import networkx as nx


# Genère un set de Nodes qui servira à construire hnsw, et un set de requetes
# qui serviront de recherches approximées
def generation(nb_nodes,nb_requests):
    gen_node = lambda : gen_random_vector(max_value, dim)

    base = set()
    while len(base) != nb_nodes :
        base.add(gen_node())

    requests = set()
    while len(requests) != nb_requests:
        node = gen_node()
        if node not in base:
            requests.add(node)

    return list(requests), list(base)

def build_hnsw(nodes):
    return HNSW(dim,M,Mmax,mL, efC, nodes)


# Construis la liste des distance entre les points du niveau 0, distance
# obtenue par shortest path sur graphe de leakage
def construction_listekage(Base,Requetes):
    K  = 1 # Nb of nearest neighbours
    ef = 1 # Nb of neighbours to consider

    Base = Known_q.permutation_aleatoire(Base)
    hnsw = build_hnsw(Base)
    paths_to_leaves = []
    mimosa = {}
    undirected_mimosa = {}
    graph_leakage = {}

    for q in Base :
        result, path = hnsw.knn_search(q, K, ef,[])
        if result != [q] :
            return construction_listekage(Base, Requetes)
        else:
            paths_to_leaves.append(path)

    for q in Requetes:
        _, path = hnsw.knn_search(q, K, ef, [])
        paths_to_leaves.append(path)

    mimosa = arbre.arborize(paths_to_leaves)
    undirected_mimosa = arbre.directed_to_undirected(mimosa)
    graph_leakage = nx.DiGraph(undirected_mimosa)
    get_dist = lambda n, q : nx.shortest_path_length(graph_leakage,(q,0),(n,0))
    return [[get_dist(n, q) for n in Base] for q in Base]


# Construis la liste des distances entre les points du niveau 0, distance
# obtenue par shortest path sur graphe de Delaunay
def Construction_Distlaunay(Base, dim):
    G_0 = rdc.Set_to_DLGraph(Base, dim)
    get_dist = lambda n, q : nx.shortest_path_length(G_0,q,n)
    return [[get_dist(n, q) for n in Base] for q in Base]

# Resort le nombre de keywords retrouvé. (Test est le nombre de construction
# d'hnsw que l'attaquant fait pour construire le dictionnaire de score. explose
# la complexité, à mettre en pré-calcul pour faire des stats)
def score(test, nb_requetes, nb_nodes,known, dim):
    score_moyen = {}
    Requetes, Base = generation(nb_nodes,nb_requetes)
    L_distance_nodes = Construction_Distlaunay(Base, dim)
    L_distance_graph = []
    for _ in range (test):                                              # Construis le dictionnaire de score
        L_distance_graph = construction_listekage(Base,Requetes)
        score = Known_q.scoring(L_distance_nodes,L_distance_graph)
        score_moyen = Known_q.somme_deux_dictionnaires(score_moyen,score)

    results = Known_q.Scoring_association(known,L_distance_nodes,L_distance_graph,score_moyen)
    wins = Known_q.evaluation(results)
    return wins


if __name__ == "__main__":
    dim = 3                     #dimension des vecteurs
    M = 8                       #Nombre de connexion lors de l'insertion
    Mmax = 8                    #Nombre de connexion maximale d'un noeuds dans hnsw (Mmax=2*Mmax au niveau 0)
    mL = 0.9                    #Coefficient qui pondere la loi de proba (plus mL est haut et plus le niveau max est haut)
    max_value = 100             #Max value pour les coefficient du vecteur
    nb_nodes = 100              #Nombre de vecteurs dans hnsw
    nb_requetes = nb_nodes*10          #Nombre de requetes approximées
    efC = 10                    #Coefficient de voisins explorés lors de la construction
    known = 10                  #Nombre de vecteurs connus

    score_value = score(10,nb_requetes,nb_nodes,known, dim)

    print('nombre de keywords trouvé :', score_value, 'en en connaissant', known, 'sur', nb_nodes)

