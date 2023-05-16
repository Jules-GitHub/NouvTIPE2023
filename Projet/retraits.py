import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from tqdm import tqdm
import os

def distance(points, point):
    return min([((x - point[0]) ** 2 + (y - point[1]) ** 2) ** 0.5 for x, y in points])

def affichage(points, zone, e):
    plt.axis(zone)
    mini = np.inf
    maxi = -np.inf
    xMax = -1
    yMax = -1
    valeurs = np.zeros((int((zone[1] - zone[0])/e), int((zone[3] - zone[2])/e)))

    i=0
    for x in range(int(zone[0]/e), int(zone[1]/e)):
        j=0
        for y in range(int(zone[2]/e), int(zone[3]/e)):
            print(x, y)
            valeurs[i][j] = distance(points, (x*e, y*e))
            if valeurs[i][j] > maxi:
                maxi = valeurs[i][j]
                xMax = x
                yMax = y
            if valeurs[i][j] < mini:
                mini = valeurs[i][j]
            j+=1
        i+=1
    
    def couleur(z):
        return (z - mini) / (maxi - mini)
    
    i=0
    for x in range(int(zone[0]/e), int(zone[1]/e)):
        j=0
        for y in range(int(zone[2]/e), int(zone[3]/e)):
            c = couleur(valeurs[i][j])
            plt.plot(x*e, y*e, '.', markersize=4, color=(c, 2.5*c*(1-c), 1 - c))
            j+=1
        i+=1
    
    plt.plot([x for x, y in points], [y for x, y in points], 'ko')
    plt.plot(xMax*e, yMax*e, 'ro')

def dist_moyenne(points, zone):
    def f(x, y):
        return distance(points, (x, y))
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
    def distance2(points, attracteurs, point):
        mini = min([((x - point[0]) ** 2 + (y - point[1]) ** 2) ** 0.5 for x, y in points])
        for x, y, coeff in attracteurs:
            d = ((x - point[0]) ** 2 + (y - point[1]) ** 2) ** 0.5
            mini *= 1 + coeff/(1+d)
        return mini
    def f(x, y):
        return distance2(points, attracteurs, (x, y))
    return integrate.dblquad(f, zone[0], zone[1], lambda _: zone[2], lambda _: zone[3])[0]/((zone[1] - zone[0]) * (zone[3] - zone[2]))

def avec_attracteur(n, nbAtt, zone, attracteurs=None, e=0.04):
    strZone = " ".join([str(x) for x in zone])
    if attracteurs is None:
        attracteurs = []
        for _ in range(nbAtt):
            attracteurs.append([np.random.uniform(zone[0], zone[1]), np.random.uniform(zone[2], zone[3]), np.random.uniform(0, 1)])
    strAttracteurs = ""
    for x, y, coeff in attracteurs:
        strAttracteurs += str(x) + " " + str(y) + " " + str(coeff) + " "
    strAttracteurs = strAttracteurs[:-1]

    sortie = os.popen("gcc attracteur.c -O2 && ./a.out " + str(n) + " " + str(nbAtt) + " " + strZone + " " + strAttracteurs).read()
    points = delinearise(list(map(float, sortie.split(" ")[:-1])))

    affichage(points, zone, e)
    plt.plot([x for x, _, _ in attracteurs], [y for _, y, _ in attracteurs], 'k+', markersize=30)

    print("Points :", points)
    print("Attracteur :", attracteurs)
    print("Distance moyenne :", dist_moyenne2(points, attracteurs, zone))

    plt.show()

nbAtt = 3
attracteurs = [[2, 1, 0.5], [3, 2, 0.5], [2, 2, 0.5]]
avec_attracteur(5, nbAtt, [0, 4, 0, 4], attracteurs, 0.04)