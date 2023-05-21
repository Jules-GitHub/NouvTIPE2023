import matplotlib.pyplot as plt
import numpy as np
import os
from math import sqrt, exp
import pandas as pd

decoupage = 100

def calcul_coeff(nbAtt, attracteurs=None, e=0.01):
    if attracteurs is None:
        attracteurs = []
        for _ in range(nbAtt):
            attracteurs.append((np.random.uniform(-1, 1), np.random.uniform(-1, 1), 10*(1+np.random.uniform(0, 1))))
    
    strAttracteurs = ""
    for x, y, coeff in attracteurs:
        print(x, y, coeff)
        strAttracteurs += str(x) + " " + str(y) + " " + str(coeff) + " "
    strAttracteurs = strAttracteurs[:-1]

    sortie = os.popen("gcc coefficients.c -O2 -lm && ./a.out " + str(nbAtt) + " " + strAttracteurs).read()
    coeffLin = list(map(float, sortie.split(" ")[:-1]))

    coeff = np.zeros((decoupage+1, decoupage+1))
    for i in range(decoupage+1):
        for j in range(decoupage+1):
            coeff[i, j] = coeffLin[i*(decoupage+1)+j]

    return coeff

def affichage(coeff, zone):
    #Affiche les points et les coeffs en 3D
    X = np.linspace(zone[0], zone[1], decoupage+1)
    Y = np.linspace(zone[2], zone[3], decoupage+1)
    X, Y = np.meshgrid(X, Y)
    coeff.reshape(X.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, coeff.T, cmap='viridis', edgecolor='none')
    ax.scatter([x for x, y, coeff in attracteurs], [y for x, y, coeff in attracteurs], [coeff*exp(-1/sqrt(2)**5) for x, y, coeff in attracteurs], color='r')
    for i in range(len(attracteurs)):
        ax.text(attracteurs[i][0], attracteurs[i][1], attracteurs[i][2]*exp(-1/sqrt(2)**5), str(i+1), color='k')
        print(str(i+1), attracteurs[i][0], attracteurs[i][1], attracteurs[i][2]*exp(-1/sqrt(2)**5), coeff[int((attracteurs[i][0]-zone[0])/(zone[1]-zone[0])*decoupage), int((attracteurs[i][1]-zone[2])/(zone[3]-zone[2])*decoupage)])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Coeff')
    ax.set_title('Coefficients')
    print(np.max(coeff), np.argmax(coeff))
    i = np.argmax(coeff)//(decoupage+1)
    j = np.argmax(coeff)%(decoupage+1)
    print(i, j)

def combine(revenu, densite):
    return revenu/1000 * densite/100

def importe_donnees():
    posFichier = pd.read_csv("../Metz/positions.csv", sep=";")
    revenuFichier = pd.read_csv("../Metz/revenus.csv", sep=";")
    densiteFichier = pd.read_csv("../Metz/densite.csv", sep=";")
    attracteurs = [(posFichier["X"][i], posFichier["Y"][i], combine(revenuFichier["MONTANT"][i], densiteFichier["DENSITE"][i])) for i in range(len(posFichier))]
    return attracteurs, len(attracteurs)

attracteurs, nbAtt = importe_donnees()

affichage(calcul_coeff(nbAtt, attracteurs), [0, 1, 0, 1])
plt.show()