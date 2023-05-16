import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import os
import pandas as pd
from math import exp, sqrt

zone = (0, 1, 0, 1)

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

def dist_moyenne(points, attracteurs):
    def f(x, y):
        return distance(points, attracteurs, (x, y))
    return integrate.dblquad(f, 0, 1, lambda _: 0, lambda _: 1)[0]

def affichage(X, Y, Z, attracteurs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(zone[0], zone[1])
    ax.set_ylim3d(zone[2], zone[3])

    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    maxi = np.max([coeff for x, y, coeff in attracteurs]) if len(attracteurs)>0 else 1
    mini = np.min(Z)
    ax.scatter([x for x, y, coeff in attracteurs], [y for x, y, coeff in attracteurs], [coeff/maxi*mini for x, y, coeff in attracteurs], color='r')
    for i in range(len(attracteurs)):
        ax.text(attracteurs[i][0], attracteurs[i][1], attracteurs[i][2]/maxi*mini, str(i+1), color='k')

def variation_position(attracteurs, e=0.01):
    strAttracteurs = ""
    for x, y, coeff in attracteurs:
        strAttracteurs += str(x) + " " + str(y) + " " + str(coeff) + " "
    strAttracteurs = strAttracteurs[:-1]
    scores = []
    X = np.arange(0, 1, e)
    Y = np.arange(0, 1, e)
    os.popen("gcc score.c -O2")
    for x in X:
        print(x)
        scores.append([])
        for y in Y:
            score = os.popen("./a.out 1 " + str(len(attracteurs)) + " " + str(x) + " " + str(y) + " " + strAttracteurs).read()
            scores[-1].append(float(score))
    X, Y = np.meshgrid(X, Y)
    scores = np.reshape(scores, X.shape)
    i,j = np.argmin(scores)//len(scores), np.argmin(scores)%len(scores)
    print(X[i][j], Y[i][j], scores[i][j])
    affichage(X, Y, scores, attracteurs)
    plt.show()

def importe_donnees():
    posFichier = pd.read_csv("../positions.csv", sep=";")
    poidsFichier = pd.read_csv("../revenus.csv", sep=";")
    attracteurs = [(posFichier["X"][i], posFichier["Y"][i], poidsFichier["MONTANT"][i]/100) for i in range(len(posFichier))]
    return attracteurs, len(attracteurs)

attracteurs, nbAtt = importe_donnees()

variation_position(attracteurs)

"""
strAttracteurs = ""
for x, y, coeff in attracteurs:
    strAttracteurs += str(x) + " " + str(y) + " " + str(coeff) + " "
strAttracteurs = strAttracteurs[:-1]

print("./a.out 1 " + str(nbAtt) + " " + str(0) + " " + str(0) + " " + strAttracteurs)
"""