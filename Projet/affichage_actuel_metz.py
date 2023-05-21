import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import pandas as pd
from math import sqrt, exp

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

def affichage(points, zone, e):
    plt.axis(zone)
    mini = np.inf
    maxi = -np.inf
    xMax = -1
    yMax = -1
    valeurs = np.zeros((int((zone[1] - zone[0])/e), int((zone[3] - zone[2])/e)))

    for x in range(int(zone[0]/e), int(zone[1]/e)):
        for y in range(int(zone[2]/e), int(zone[3]/e)):
            valeurs[x][y] = distance(points, (x*e, y*e))
            if valeurs[x][y] > maxi:
                maxi = valeurs[x][y]
                xMax = x
                yMax = y
            if valeurs[x][y] < mini:
                mini = valeurs[x][y]
    
    def couleur(z):
        return (z - mini) / (maxi - mini)
    
    image = np.zeros((int((zone[1] - zone[0])/e), int((zone[3] - zone[2])/e), 4))
    for x in range(int(zone[0]/e), int(zone[1]/e)):
        for y in range(int(zone[2]/e), int(zone[3]/e)):
            c = couleur(valeurs[x][y])
            image[y][x] = (c, 2.5*c*(1-c), 1 - c, 0.9)

    plt.imshow(image, extent=zone)
    
    plt.plot([x for x, y in points], [y for x, y in points], 'ko')
    plt.plot(xMax*e, yMax*e, 'ro')
    plt.plot([x for x, _, _ in attracteurs], [y for _, y, _ in attracteurs], 'k+', markersize=30)

def affichage2(points, attracteurs, zone, e):
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
    
    """
    image = np.zeros((int((zone[3] - zone[2])/e), int((zone[1] - zone[0])/e), 4))
    for x in range(int(zone[0]/e), int(zone[1]/e)):
        for y in range(int(zone[2]/e), int(zone[3]/e)):
            c = couleur(valeurs[x][y])
            image[y][x] = (c, 2.5*c*(1-c), 1 - c, 0.9)

    plt.imshow(np.flip(image, 0), extent=zone)
    """

    for x in range(int(zone[0]/e), int(zone[1]/e)):
        for y in range(int(zone[2]/e), int(zone[3]/e)):
            c = couleur(valeurs[x][y])
            plt.plot(x*e, y*e, '.', markersize=4, color=(c, 2.5*c*(1-c), 1 - c))
    
    plt.plot([x for x, y in points], [y for x, y in points], 'ko')
    plt.plot(xMax*e, yMax*e, 'ro')
    plt.plot([x for x, _, _ in attracteurs], [y for _, y, _ in attracteurs], 'k+', markersize=20)
    for i in range(len(attracteurs)):
        plt.text(attracteurs[i][0], attracteurs[i][1], str(i+1), color='k', fontsize=10)

def affichage3d(points, attracteurs, zone, e):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(zone[0], zone[1])
    ax.set_ylim3d(zone[2], zone[3])
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
    
    X, Y = np.meshgrid(np.arange(zone[0], zone[1], e), np.arange(zone[2], zone[3], e))
    Z = np.reshape(valeurs, X.shape)

    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    
    ax.scatter([x for x, y in points], [y for x, y in points], 0, c='k')
    ax.scatter(xMax*e, yMax*e, 0, c='r')
    plt.plot([x for x, _, _ in attracteurs], [y for _, y, _ in attracteurs], 'k+', markersize=30)

def dist_moyenne2(points, attracteurs, zone):
    def f(x, y):
        return distance2(points, attracteurs, (x, y))
    return integrate.dblquad(f, zone[0], zone[1], lambda _: zone[2], lambda _: zone[3])[0]/((zone[1] - zone[0]) * (zone[3] - zone[2]))


def combine(revenu, densite):
    return revenu/1000 * densite/100

def importe_donnees():
    posFichier = pd.read_csv("../Metz/positions.csv", sep=";")
    revenuFichier = pd.read_csv("../Metz/revenus.csv", sep=";")
    densiteFichier = pd.read_csv("../Metz/densite.csv", sep=";")
    attracteurs = [(posFichier["X"][i], posFichier["Y"][i], combine(revenuFichier["MONTANT"][i], densiteFichier["DENSITE"][i])) for i in range(len(posFichier))]
    return attracteurs, len(attracteurs)

def importe_points():
    retraitsFichier = pd.read_csv("../Metz/retraits_actuels.csv", sep=";")
    points = [(retraitsFichier["X"][i], retraitsFichier["Y"][i]) for i in range(len(retraitsFichier))]
    return points

attracteurs, nbAtt = importe_donnees()

points = importe_points()

score = dist_moyenne2(points, attracteurs, [0, 1, 0, 1])

print("Distance moyenne :", score)

affichage2(points, attracteurs, [0, 1, 0, 1], 0.01)
plt.show()
