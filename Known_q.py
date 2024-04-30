import random

def inverse_dictionnaire(dictionnaire):
    dictionnaire_inverse = {}
    for cle, valeur in dictionnaire.items():
        if valeur in dictionnaire_inverse:
            dictionnaire_inverse[valeur].append(cle)
        else:
            dictionnaire_inverse[valeur] = [cle]
    return dict(sorted(dictionnaire_inverse.items(), reverse=True))

def Scoring_association(known, Delaunay,Leakage, score ): # All inclusive, retourne un score pour chaque association. Ne l'interprete pas.
    F=[]
    for lbd in range(known):# Pour chaque knownQ donne un score à chacune des unknown querries O(n^2)
        F.append([])
        for i in range (known,len(Delaunay[lbd])):
            for j in range(known,len(Leakage[lbd])):
                u=(Delaunay[lbd][i],Leakage[lbd][j])
                s=0
                if u in score:
                    s=score[u]
                else:
                    s=-100
                F[lbd].append(((i,j),s))

    sum_score_association = {}
    # Calcul de la somme des scores pour chaque coordonnée
    for sous_liste in F:
        for coordonnee, score in sous_liste:
            if coordonnee in sum_score_association:
                sum_score_association[coordonnee] += score
            else:
                sum_score_association[coordonnee] = score
    return inverse_dictionnaire(sum_score_association)


def evaluation(Dico):
    L=[]
    MissLeft=set()
    MissRight=set()
    for i in Dico.keys() :
        for j in Dico[i]:
            if j[0]==j[1]:
                if j[0] not in MissLeft and j[0] not in MissRight:
                    L.append(j[0])

            else:
                if j[0] not in L and j[1] not in L:
                    if j[0] not in MissLeft and j[1] not in MissRight:
                        
                        MissLeft.add(j[0])
                        MissRight.add(j[1]) #traiter comme deux listes (1,2)/!(2,1)----> faire deux listes miss
    return len(L)



def compter_occurrences(liste_de_listes):
    occurrences = {}
    for sous_liste in liste_de_listes:
        for nombre in sous_liste:
            if nombre in occurrences:
                occurrences[nombre] += 1
            else:
                occurrences[nombre] = 1
    return occurrences

def scoring(Delaunay,Leakage):     #(Delaunay,Leakage), nbr de permutation pour l'entrainement.
    score={}
    Fscore={}
    for lbd in range(len(Leakage)):
        for i in range (len(Delaunay[lbd])):
            if (Delaunay[lbd][i],Leakage[lbd][i]) in score :
                score[(Delaunay[lbd][i],Leakage[lbd][i])]+=1
            else:
                score[(Delaunay[lbd][i],Leakage[lbd][i])]=1
            
            for j in range(len(Delaunay)):
                if (Delaunay[lbd][i],Leakage[lbd][j]) in Fscore :
                    Fscore[(Delaunay[lbd][i],Leakage[lbd][j])]+=1
                else:
                    Fscore[(Delaunay[lbd][i],Leakage[lbd][j])]=1
    for couple in score:
        score[couple]=int(score[couple]/Fscore[couple]*1000)

    #print(inverse_dictionnaire(Fscore))
    #print(inverse_dictionnaire(score))
    return score
    

def somme_deux_dictionnaires(dict1, dict2):
    resultat = {}

    # Parcourir les clés et les valeurs du premier dictionnaire
    for cle, valeur in dict1.items():
        # Si la clé existe dans les deux dictionnaires, ajouter les valeurs ensemble
        if cle in dict2:
            resultat[cle] = valeur + dict2[cle]
        # Si la clé n'existe que dans le premier dictionnaire, ajouter la paire clé/valeur au résultat
        else:
            resultat[cle] = valeur

    # Ajouter les paires clé/valeur restantes du deuxième dictionnaire au résultat
    for cle, valeur in dict2.items():
        if cle not in resultat:
            resultat[cle] = valeur

    return resultat


def permutation_aleatoire(liste):
    # Copier la liste originale pour éviter de la modifier directement
    liste_permutee = liste.copy()
    # Permuter aléatoirement la liste
    random.shuffle(liste_permutee)
    return liste_permutee



