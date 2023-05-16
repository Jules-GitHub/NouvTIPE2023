import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from math import exp, sqrt

def distance(points, attracteurs, point):
    mini = min([((x - point[0]) ** 2 + (y - point[1]) ** 2) ** 0.5 for x, y in points])
    dist = float('inf')
    coefficient = 0
    for x, y, coeff in attracteurs:
        d = ((x - point[0]) ** 2 + (y - point[1]) ** 2) ** 0.5
        if d < dist:
            dist = d
            coefficient = coeff*exp(-1/(sqrt(2)-dist)**5)
    mini *= 1 + coefficient
    return mini

def affichage(points, attracteurs, zone, e):
    plt.axis(zone)
    mini = np.inf
    maxi = -np.inf
    xMax = -1
    yMax = -1
    valeurs = np.zeros((int((zone[1] - zone[0])/e), int((zone[3] - zone[2])/e)))

    for x in range(int(zone[0]/e), int(zone[1]/e)):
        for y in range(int(zone[2]/e), int(zone[3]/e)):
            valeurs[x][y] = distance(points, attracteurs, (x*e, y*e))
            print(x, y, valeurs[x][y])
            if valeurs[x][y] > maxi:
                maxi = valeurs[x][y]
                xMax = x
                yMax = y
            if valeurs[x][y] < mini:
                mini = valeurs[x][y]
    
    def couleur(z):
        return (z - mini) / (maxi - mini)
    
    for x in range(int(zone[0]/e), int(zone[1]/e)):
        for y in range(int(zone[2]/e), int(zone[3]/e)):
            c = couleur(valeurs[x][y])
            plt.plot(x*e, y*e, '.', markersize=4, color=(c, 2.5*c*(1-c), 1 - c))
    
    plt.plot([x for x, y in points], [y for x, y in points], 'ko')

    plt.plot([x for x, _, _ in attracteurs], [y for _, y, _ in attracteurs], 'k+', markersize=20)
    plt.plot(xMax*e, yMax*e, 'ro')

def delinearise(x):
    n = len(x)//2
    return np.array([[x[2*i], x[2*i+1]] for i in range(n)])

def avec_attracteur(n, nbAtt, zone, attracteurs=None, e=0.01):
    strZone = " ".join([str(x) for x in zone])
    if attracteurs is None:
        attracteurs = []
        for _ in range(nbAtt):
            attracteurs.append([np.random.uniform(zone[0], zone[1]), np.random.uniform(zone[2], zone[3]), 100*(1+np.random.uniform(0, 1))])
    strAttracteurs = ""
    for x, y, coeff in attracteurs:
        strAttracteurs += str(x) + " " + str(y) + " " + str(coeff) + " "
    strAttracteurs = strAttracteurs[:-1]

    print("Attracteurs :", attracteurs)

    sortie = os.popen("gcc presentation_avec_att.c -O2 && ./a.out " + str(n) + " " + str(nbAtt) + " " + strZone + " " + strAttracteurs).read()
    sortie2 = list(map(float, sortie.split(" ")))

    points = delinearise(sortie2[:-1])
    score = sortie2[-1]

    affichage(points, attracteurs, zone, e)

    print("Points :", points)
    print("Distance moyenne :", score)
    plt.show()

def importe_donnees():
    posFichier = pd.read_csv("../positions.csv", sep=";")
    poidsFichier = pd.read_csv("../revenus.csv", sep=";")
    attracteurs = [(posFichier["X"][i], posFichier["Y"][i], poidsFichier["MONTANT"][i]/100) for i in range(len(posFichier))]
    return attracteurs, len(attracteurs)

attracteurs, nbAtt = importe_donnees()
avec_attracteur(10, nbAtt, [0, 1, 0, 1], attracteurs, e=0.01)