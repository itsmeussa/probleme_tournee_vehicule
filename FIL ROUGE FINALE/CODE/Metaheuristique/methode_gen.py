
#Voici un exemple d'implémentation de l'algorithme génétique pour résoudre le problème de tournée de véhicules. La fonction d'initialisation crée une population de solutions aléatoires, la fonction de sélection choisit les solutions les plus aptes pour la reproduction, la fonction de reproduction crée de nouvelles solutions à partir de celles sélectionnées, la fonction de mutation modifie aléatoirement certaines solutions, et la fonction d'évaluation calcule la qualité de chaque solution. Le processus de sélection-reproduction-mutation-évaluation est répété plusieurs fois jusqu'à ce qu'une solution satisfaisante soit trouvée.

"""
Problème de tournée de véhicules avec l'algorithme génétique
"""

from copy import deepcopy
from math import *
import random as rd
import os
import pandas as pd
import geopy.distance
import matplotlib.pyplot as plt
Q = 440 # capacité d’un camion
nbitermax = 10000 # nb d'itérations maximal pour la fonction recuit
nb_populations = 100 # nombre de solutions dans la population
nb_generations = 10 # nombre de générations pour l'algorithme génétique

def getPoids():
    # Calcul du poids pour chaque customer, on prend le poids max
    df = pd.read_excel("./DATASET/2_detail_table_customers.xls")
    df[["CUSTOMER_NUMBER", "TOTAL_WEIGHT_KG"]]
    poids = [0 for i in range(131)]
    for id, row in df.iterrows():
        customer_id = row["CUSTOMER_NUMBER"]
        weight = row["TOTAL_WEIGHT_KG"]
        if poids[customer_id] < weight:
            poids[customer_id] = weight
    return poids

def getDistance():
    distance = [[0 for _ in range(131)] for _ in range(131)]
    df_depot = pd.read_excel("./DATASET/6_detail_table_cust_depots_distances.xls")
    df_depot = df_depot[["CUSTOMER_NUMBER","DISTANCE_KM", "DIRECTION"]]

    df_customer = pd.read_excel("./DATASET/2_detail_table_customers.xls")
    df_customer = df_customer[df_customer["ROUTE_ID"] == 2970877]
    df_customer = df_customer[["CUSTOMER_NUMBER","CUSTOMER_LATITUDE","CUSTOMER_LONGITUDE"]]


    # Ajout distance pour dépôt
    for id, row in df_depot.iterrows():
        if row["DIRECTION"] == "DEPOT->CUSTOMER":
            distance[0][row["CUSTOMER_NUMBER"]] = row["DISTANCE_KM"]
        else :
            distance[row["CUSTOMER_NUMBER"]][0] = row["DISTANCE_KM"]

    # Ajout distance entre chaque customer
    for id, row1 in df_customer.iterrows():
        for id, row2 in df_customer.iterrows():
            i = int(row1["CUSTOMER_NUMBER"])
            coord_i = (row1["CUSTOMER_LATITUDE"],row1["CUSTOMER_LONGITUDE"])
            j = int(row2["CUSTOMER_NUMBER"])
            coord_j = (row2["CUSTOMER_LATITUDE"], row2["CUSTOMER_LONGITUDE"])

            if i != j:
                distance[i][j] = geopy.distance.geodesic(coord_i, coord_j).km

    return distance

"""Création du problème"""
def creationdemande():
    # Cette fonction crée le problème avec les poids des arêtes et les quantités de marchandises
    # Elle retourne une matrice L qui contient les poids des arêtes et les quantités de marchandises
    poids = getPoids()
    distance = getDistance()
    L = []
    M = [0] # Le sommet 0 correspond à l'entrepôt où il n'y a pas de commande
    for j in range(131): # Il y a 130 lieux de livraison numérotés de 1 à 131
        d = distance[0][j] # Distance entre entrepot et sommet j
        M.append(d)
    L.append(M)
    for i in range(131):
        M = []
        q = poids[i] # Générer une quantité de marchandise pour le sommet i
        M.append(q)
        for j in range(131):
            d = distance[i][j] #Distance entre les sommets i et j
            M.append(d)
        L.append(M)

    return L

def nbcamions(L):
    # Cette fonction calcule le nombre de camions nécessaires pour livrer toutes les marchandises
    Qtot = 0 # La quantité de marchandise pour un tour
    for i in range(len(L)):
        Qtot = Qtot + L[i][0]
    nbcamions = ceil(Qtot / Q) # Arrondi supérieur
    return nbcamions + 6


# Fonction auxiliaire pour calculer le coût d'une solution
def coutsolution(L, s, K):
    cout = 0
    for k in range(K):
        for i in range(len(s[k]) - 1):
            cout = cout + L[s[k][i]][s[k][i+1]]
        cout = cout + L[s[k][-1]][0] # retour à l'entrepôt
    return cout

# Fonction d'opérateur de croisement en utilisant l'algorithme de crossover en deux points
def croisement(a, b, K):
    s1 = deepcopy(a)
    s2 = deepcopy(b)
    n = len(s1[0])
    p1 = rd.randint(1, n-2) # premier point de croisement
    p2 = rd.randint(p1, n-1) # deuxième point de croisement
    for k in range(K):
        t = s1[k][p1:p2]
        s1[k][p1:p2] = [x for x in s2[k] if x not in t]
        s2[k][p1:p2] = [x for x in a[k] if x not in t]
    return s1, s2



"""Algorithme génétique"""

def initialisation(L,K): # création de la population initiale
    pop = []
    for i in range(nb_populations):
        s = [[] for k in range(K)]
        for k in range(K):
            s[k] = rd.sample(range(1, len(L)), len(L)-1) # choix aléatoire des sommets à visiter pour chaque camion
        pop.append(s)
    return pop

def selection(L, pop, K): # sélection des solutions les plus aptes
    couts = []
    for s in pop:
        couts.append(coutsolution(L, s, K))
    pop_couts = list(zip(pop, couts))
    pop_couts.sort(key=lambda x: x[1]) # tri des solutions selon leur coût croissant
    meilleurs = [x[0] for x in pop_couts[:int(nb_populations/2)]] # les meilleures solutions sont sélectionnées
    return meilleurs

def reproduction(L, elite, K):
    enfants = []
    pop_couts = []
    # Croisement des solutions sélectionnées
    for i in range(int(nb_populations / 2)):
        a = elite[i]
        b = elite[rd.randint(0, len(elite)-1)]
        enfant1, enfant2 = croisement(a, b, K)
        enfants.append(enfant1)
        enfants.append(enfant2)
    # Calcul du coût des nouveaux enfants
    couts = [coutsolution(L, enfant, K) for enfant in enfants]
    # Ajout des nouveaux enfants à la population
    pop_couts += list(zip(enfants, couts))
    pop_couts.sort(key=lambda x: x[1])
    nouvelle_pop = [pop_couts[i][0] for i in range(nb_populations)]
    return nouvelle_pop


def mutation(pop, pmutation):
    for i in range(len(pop)):
        for k in range(len(pop[i])):
            if rd.random() < pmutation:
                a = rd.randint(0, len(pop[i][k])-1)
                b = rd.randint(0, len(pop[i][k])-1)
                pop[i][k][a], pop[i][k][b] = pop[i][k][b], pop[i][k][a]
    return pop

def evaluation(L, pop, K): # calcul de la qualité de chaque solution
    couts = []
    for s in pop:
        couts.append(coutsolution(L, s, K))
    return couts

"""Programme principal"""

def main():
    couts_tot=[]
    L = creationdemande()
    K = nbcamions(L) - 1
    pop = initialisation(L,K)
    meilleur_cout = float('inf')
    meilleure_solution = None
    for i in range(nb_generations):
        pop = selection(L, pop, K)
        pop = reproduction(L, pop, K)
        pop = mutation(pop, 0.1) # taux de mutation = 10%
        couts = evaluation(L, pop, K)
        min_cout = min(couts)
        if min_cout < meilleur_cout:
            meilleur_cout = min_cout
            meilleure_solution = pop[couts.index(min_cout)]
        couts_tot.append(meilleur_cout)
        print("Génération {}, coût minimal : {}".format(i+1, meilleur_cout))
    iterations=[i for i in range(1,11)]
    plt.plot(iterations,couts_tot)
    plt.xlabel("Nombre d'itérations")
    plt.ylabel("Cout simulé")
    plt.show()
    print("Coût minimal trouvé : {}".format(meilleur_cout))
    print("Meilleure solution trouvée : {}".format(meilleure_solution))


main()