import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from tqdm import tqdm
from math import exp, sqrt
import os
import pandas as pd

def distance(points, attracteurs, point):
    mini = min([((x - point[0]) ** 2 + (y - point[1]) ** 2) ** 0.5 for x, y in points])
    for x, y, coeff in attracteurs:
        d = ((x - point[0]) ** 2 + (y - point[1]) ** 2) ** 0.5
        mini *= 1 + coeff/(1+d)
    return mini

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

def dist_moyenne(points, attracteurs, zone):
    def f(x, y):
        return distance2(points, attracteurs, (x, y))
    return integrate.dblquad(f, zone[0], zone[1], lambda _: zone[2], lambda _: zone[3])[0]/((zone[1] - zone[0]) * (zone[3] - zone[2]))

def dist_moyenne2(points, attracteurs, zone):
    def distance3(points, attracteurs, point):
        mini = min([((x - point[0]) ** 2 + (y - point[1]) ** 2) ** 0.5 for x, y in points])
        dist = np.inf
        c = 0
        for x, y, coeff in attracteurs:
            d = ((x - point[0]) ** 2 + (y - point[1]) ** 2) ** 0.5
            if d < dist:
                dist = d
                c = coeff/(1+d)
        mini *= 1 + c
        return mini
    def f(x, y):
        return distance3(points, attracteurs, (x, y))
    return integrate.dblquad(f, zone[0], zone[1], lambda _: zone[2], lambda _: zone[3])[0]/((zone[1] - zone[0]) * (zone[3] - zone[2]))

def importe_donnees():
    posFichier = pd.read_csv("../positions.csv", sep=";")
    poidsFichier = pd.read_csv("../revenus.csv", sep=";")
    attracteurs = [(posFichier["X"][i], posFichier["Y"][i], poidsFichier["MONTANT"][i]/100) for i in range(len(posFichier))]
    return attracteurs, len(attracteurs)

#attracteurs = [(0.4925592131027712, 0.5630987065658729, 329.0), (0.530786055572615, 0.6146660481920735, 284.34), (0.6207554528526875, 0.5564552548130165, 304.89), (0.6080766496551417, 0.4721468288953031, 313.04), (0.5463408557683225, 0.3433093053870272, 331.55), (0.4576473377141903, 0.4127275501404667, 407.79), (0.3612641831240012, 0.4745092853274842, 438.04), (0.3499308425497687, 0.6822066025445567, 425.31), (0.4736903172312116, 0.7231019762394516, 325.32), (0.619998261583099, 0.7014176180069346, 231.05), (0.7304986947974371, 0.5074065749773988, 250.37), (0.7874169584075171, 0.3218510211017206, 269.47), (0.6099680591687346, 0.1476004891782399, 226.7), (0.4063006196999938, 0.1882579162342022, 270.97), (0.2438831320723534, 0.3119051952105546, 309.31), (0.1395850331396983, 0.5381191582400513, 405.32), (0.3163540725224547, 0.8212373079428296, 296.83), (0.5202446242198251, 0.8739981828527402, 189.17), (0.7514391660802316, 0.8076706837057039, 168.87), (0.8451074565150596, 0.5597884854239198, 189.53)]

attracteurs, _ = importe_donnees()
attracteurs = []

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('click to build polygon points')
ax.axis([0, 1, 0, 1])

points = []
points = plt.ginput(show_clicks=True, timeout=0, n=4)

plt.close(fig)

print(points)
affichage(points, attracteurs, (0, 1, 0, 1), 0.01)
plt.plot([x for x, _, _ in attracteurs], [y for _, y, _ in attracteurs], 'k+', markersize=30)
print(dist_moyenne(points, attracteurs, (0, 1, 0, 1)))
plt.show()