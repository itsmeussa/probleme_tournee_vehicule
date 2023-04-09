import random
import pandas as pd
import geopy.distance
import matplotlib.pyplot as plt
def getDistance():
    distance = [[0 for _ in range(131)] for _ in range(131)]
    df_depot = pd.read_excel("Metaheuristique/data_PTV_Fil_rouge/6_detail_table_cust_depots_distances.xls")
    df_depot = df_depot[["CUSTOMER_NUMBER","DISTANCE_KM", "DIRECTION"]]

    df_customer = pd.read_excel("Metaheuristique/data_PTV_Fil_rouge/2_detail_table_customers.xls")
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


#Generation de solution initiale
def initialiser_solution(nb_clients,nb_vehicules):
    # Génère une solution initiale aléatoire
    clients = list(range(1, nb_clients))
    random.shuffle(clients)
    solution = []
    for i in range(nb_vehicules):
        route = [0]  # Ajoute le dépôt au début de chaque route
        route += clients[i::nb_vehicules]  # Répartit les clients entre les routes
        route.append(0)  # Ajoute le dépôt à la fin de chaque route
        solution.append(route)
    return solution

# Calculer le coût d'une solution
def calculer_cout(solution,distances,w):
    nb_vehicules=len(solution)
    nb_clients = sum(len(route) - 2 for route in solution)
    nb_routes=nb_vehicules
    nb_clients_non_visite=nb_clients
    cout=0
    for route in solution :
        nb_clients_route=len(route)-2
        if nb_clients_route > 0 :
            cout=cout+distances[route[0]][route[1]]
            for i in range(nb_clients_route-1):
                cout+=distances[route[i+1]][route[i+2]]
            nb_clients_non_visite-=nb_clients_route
        else :
            nb_routes-=1
    cout=cout+nb_routes*w*nb_clients_non_visite
    return cout

#Generation de l'ensemble de solutions voisinnages
def voisinnage_de_solution(solution):
    # Génère un voisinage en échangeant deux clients de deux routes différentes
    voisinage = []
    nb_vehicules = len(solution)
    for i in range(nb_vehicules):
        for j in range(i+1, nb_vehicules):
            route1 = solution[i]
            route2 = solution[j]
            for k in range(1, len(route1)-1):
                for l in range(1, len(route2)-1):
                    voisine1 = route1[:k] + [route2[l]] + route1[k+1:-1] + [route1[-1]]
                    voisine2 = route2[:l] + [route1[k]] + route2[l+1:-1] + [route2[-1]]
                    voisinage.append(solution[:i] + [voisine1] + solution[i+1:j] + [voisine2] + solution[j+1:])
    return voisinage

#Création de fonction d'aspiration
def fonction_aspiration(solution_initiale,distances,penalite,nb_iterations,taille_tabou):
    solution_courante = solution_initiale
    meilleur_solution = solution_courante
    meilleur_cout = calculer_cout(solution_courante, distances, penalite)
    tabou = []
    couts=[]
    for i in range(nb_iterations):
        voisins = voisinnage_de_solution(solution_courante)
        meilleur_voisin = None
        meilleur_cout_voisin = float('inf')
        for voisin in voisins:
            cout_voisin = calculer_cout(voisin, distances, penalite)
            if cout_voisin < meilleur_cout_voisin and voisin not in tabou:
                meilleur_voisin = voisin
                meilleur_cout_voisin = cout_voisin
        if meilleur_voisin is None:
            break
        solution_courante = meilleur_voisin
        if meilleur_cout_voisin < meilleur_cout:
            meilleur_solution = meilleur_voisin
            meilleur_cout = meilleur_cout_voisin
        tabou.append(meilleur_voisin)
        if len(tabou) > taille_tabou:
            tabou.pop(0)
        couts.append(meilleur_cout)
    return meilleur_solution, meilleur_cout, couts

solution_initiale=initialiser_solution(110,131)
penalite = 100
nb_iterations = 10
taille_tabou = 10
meilleure_solution, meilleur_cout,couts= fonction_aspiration(solution_initiale, getDistance(), penalite, nb_iterations, taille_tabou)
print(couts)
iterations=[i for i in range(1,11)]
print("Le meilleur cout est",meilleur_cout)
plt.plot(iterations,couts)
plt.xlabel("Nombre d'itérations")
plt.ylabel("Cout simulé")
plt.show()


