import numpy as np
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

def initialiser_solution(nb_vehicules,nb_clients):
    clients=list(range(1,nb_clients))
    random.shuffle(clients)
    solution=[]
    for i in range(nb_vehicules):
        route=[0]
        route=route+clients[i::nb_vehicules]
        route.append(0)
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

# Étape de Recuit Simulé
def recuit_simule(nb_vehicule, nb_customers, distances,T,alpha,T_min):
    # Générer une solution initiale aléatoire
    w=10
    sol_act = initialiser_solution(nb_vehicule, nb_customers)
    cout_act = calculer_cout(sol_act, distances,w)
    
    while T > T_min:
        # Générer une solution voisine
        i = random.randint(0, nb_vehicule-1)
        j, k = random.sample(range(len(sol_act[i])), 2)
        nv_sol = sol_act.copy()
        nv_sol[i][j], nv_sol[i][k] = nv_sol[i][k], nv_sol[i][j]
        cout_nv = calculer_cout(nv_sol, distances,w)
        
        # Accepter la solution si elle est meilleure ou si la probabilité de l'accepter est suffisante
        delta = cout_nv - cout_act
        if delta < 0 or np.exp(-delta / T) > random.random():
            sol_act = nv_sol
            cout_act = cout_nv
        
        # Réduire la température
        T *= alpha
    
    return sol_act, cout_act
T = 1.0
T_min = 0.00001
alpha = 0.9
nb_iterations=10
couts=[]
for i in range(nb_iterations):
    result=recuit_simule(110,131,getDistance(),T,alpha,T_min)
    print("La meilleur solution est",result[0])
    print("Le meilleur cout est",result[1])
    couts.append(result[1])
iterations=[i for i in range(1,11)]
plt.plot(iterations,couts)
plt.xlabel("Nombre d'itérations")
plt.ylabel("Cout simulé")
plt.show()