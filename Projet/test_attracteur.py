import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from tqdm import tqdm
import os
import pandas as pd
from math import exp, sqrt

def distance(points, point):
    return min([((x - point[0]) ** 2 + (y - point[1]) ** 2) ** 0.5 for x, y in points])

def distance2(points, attracteurs, point):
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
            valeurs[x][y] = distance2(points, attracteurs, (x*e, y*e))
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
    plt.plot(xMax*e, yMax*e, 'ro')

def dist_moyenne(points, zone):
    def f(x, y):
        return distance2(points, (x, y))
    return integrate.dblquad(f, zone[0], zone[1], lambda _: zone[2], lambda _: zone[3])[0]/((zone[1] - zone[0]) * (zone[3] - zone[2]))

def delinearise(x):
    n = len(x)//2
    return np.array([[x[2*i], x[2*i+1]] for i in range(n)])

def objectif(x):
    n = len(x)//2
    points = [[x[2*i], x[2*i+1]] for i in range(n)]
    return dist_moyenne(points)

def calcul_et_affiche(n, zone, e):
    strZone = " ".join([str(x) for x in zone])
    sortie = os.popen("gcc gradient.c -O2 && ./a.out " + str(n) + " " + strZone).read()
    points = delinearise(list(map(float, sortie.split(" ")[:-1])))
    affichage(points, zone, e)
    plt.show()

def minimisation(debut, fin, zone):
    X = np.arange(debut, fin, 1)
    Y = np.zeros_like(X, dtype=float)
    strZone = " ".join([str(x) for x in zone])
    os.popen("gcc minimise.c -O2")
    for i in tqdm(range(fin-debut)):
        sortie = os.popen("./a.out " + str(X[i]) + " " + strZone).read()
        score = float(sortie)
        Y[i] = score
    plt.plot(X, Y)
    plt.show()

def dist_moyenne2(points, attracteurs, zone):
    def distance3(points, attracteurs, point):
        mini = min([((x - point[0]) ** 2 + (y - point[1]) ** 2) ** 0.5 for x, y in points])
        for x, y, coeff in attracteurs:
            d = ((x - point[0]) ** 2 + (y - point[1]) ** 2) ** 0.5
            mini *= 1 + coeff/(1+d)
        return mini
    def f(x, y):
        return distance3(points, attracteurs, (x, y))
    return integrate.dblquad(f, zone[0], zone[1], lambda _: zone[2], lambda _: zone[3])[0]/((zone[1] - zone[0]) * (zone[3] - zone[2]))

def gradient(f, x, h=1e-5):
    grad = np.zeros_like(x)
    y = f(x)
    x1 = x.copy()
    for i in range(len(x)):
        x1[i] += h
        grad[i] = (f(x1) - y)/h
        x1[i] -= h
    return grad

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

    
    #print("gcc attracteur.c -O2 && ./a.out " + str(n) + " " + str(nbAtt) + " " + strZone + " " + strAttracteurs)

    print("Attracteurs :", attracteurs)

    sortie = os.popen("gcc attracteur.c -O2 && ./a.out " + str(n) + " " + str(nbAtt) + " " + strZone + " " + strAttracteurs).read()
    sortie2 = list(map(float, sortie.split(" ")))

    points = delinearise(sortie2[:-1])
    score = sortie2[-1]

    affichage(points, attracteurs, zone, e)
    plt.plot([x for x, _, _ in attracteurs], [y for _, y, _ in attracteurs], 'k+', markersize=30)

    print("Points :", points)
    print("Distance moyenne :", score)
    #print("||grad|| =", np.linalg.norm(gradient(lambda x: dist_moyenne2(x, attracteurs, zone), points)))
    #print("Gradient :", gradient(lambda x: dist_moyenne2(x, attracteurs, zone), points))"""

    plt.show()

def importe_donnees():
    posFichier = pd.read_csv("../positions.csv", sep=";")
    poidsFichier = pd.read_csv("../revenus.csv", sep=";")
    attracteurs = [(posFichier["X"][i], posFichier["Y"][i], poidsFichier["MONTANT"][i]/100) for i in range(len(posFichier))]
    return attracteurs, len(attracteurs)

attracteurs, nbAtt = importe_donnees()
avec_attracteur(1, 1, [0, 1, 0, 1], attracteurs=[(0.1, 0.1, 5)], e=0.01)