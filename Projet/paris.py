import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from tqdm import tqdm
import os
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
    
    for x in range(int(zone[0]/e), int(zone[1]/e)):
        for y in range(int(zone[2]/e), int(zone[3]/e)):
            c = couleur(valeurs[x][y])
            plt.plot(x*e, y*e, '.', markersize=4, color=(c, 2.5*c*(1-c), 1 - c))
    
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
    
    for x in range(int(zone[0]/e), int(zone[1]/e)):
        for y in range(int(zone[2]/e), int(zone[3]/e)):
            c = couleur(valeurs[x][y])
            plt.plot(x*e, y*e, '.', markersize=4, color=(c, 2.5*c*(1-c), 1 - c))
    
    plt.plot([x for x, y in points], [y for x, y in points], 'ko')
    plt.plot(xMax*e, yMax*e, 'ro')
    plt.plot([x for x, _, _ in attracteurs], [y for _, y, _ in attracteurs], 'k+', markersize=20)

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
    def f(x, y):
        return distance2(points, attracteurs, (x, y))
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

def lecture_vers_donnees(sortie):
    sortie2 = list(map(float, sortie.split(" ")))

    points = delinearise(sortie2[:-1])
    score = sortie2[-1]

    return points, score

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

    
    #print("gcc attracteur.c -O2 && ./a.out " + str(n) + " " + str(nbAtt) + " " + strZone + " " + strAttracteurs)

    #print(attracteurs)

    sortie = os.popen("gcc attracteur.c -O2 && ./a.out " + str(n) + " " + str(nbAtt) + " " + strZone + " " + strAttracteurs).read()

    points, score = lecture_vers_donnees(sortie)

    affichage2(points, attracteurs, zone, e)

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

avec_attracteur(10, nbAtt, [0, 1, 0, 1], attracteurs, 0.01)

"""
s = "0.061638 0.395368 0.184579 0.210940 0.471311 0.688862 0.492497 0.076841 0.408909 0.805923 0.912270 0.701345 0.662412 0.308757 0.420377 0.220539 0.206169 0.932888 0.738872 0.573719 0.741439 0.737498 0.075549 0.053139 0.478462 0.477759 0.330423 0.461697 0.193038 0.686808 0.298351 0.196595 0.792833 0.236515 0.331495 0.698839 0.633805 0.456624 0.064266 0.492773 0.551231 0.229331 0.533902 0.921706 0.349988 0.929998 0.087472 0.807983 0.212120 0.074120 0.168152 0.334514 0.583576 0.593177 0.247515 0.812725 0.790766 0.405343 0.197518 0.457190 0.645178 0.097718 0.062477 0.276014 0.588182 0.762458 0.057550 0.594309 0.714862 0.913644 0.406052 0.364917 0.356327 0.071160 0.396652 0.578595 0.915160 0.525061 0.899134 0.896068 0.070531 0.932469 0.278080 0.578185 0.162773 0.567326 0.286183 0.328363 0.060359 0.697586 0.069996 0.159999 0.527116 0.358145 0.928835 0.123703 0.927561 0.321901 0.788333 0.075527 11.460285"
s2 = "0.447880 0.102149 0.572355 0.047664 0.069995 0.369998 0.600500 0.649122 0.102102 0.328891 0.176990 0.979603 0.420332 0.670307 0.293246 0.969143 0.509893 0.616136 0.138177 0.780646 0.097273 0.189297 0.475532 0.264934 0.024187 0.248063 0.947671 0.281620 0.019400 0.366770 0.450606 0.341623 0.069563 0.889968 0.169775 0.885997 0.661322 0.058739 0.071960 0.796600 0.402231 0.528261 0.227848 0.862543 0.793475 0.169384 0.204520 0.217601 0.586530 0.362989 0.930869 0.806897 0.852401 0.234793 0.493067 0.869446 0.324447 0.433379 0.259996 0.550001 0.174034 0.702164 0.100193 0.280047 0.240824 0.288628 0.022864 0.935893 0.020835 0.684149 0.234601 0.724773 0.027662 0.514720 0.020165 0.442605 0.440962 0.469403 0.939476 0.169705 0.151483 0.600672 0.942623 0.389355 0.011723 0.541403 0.107376 0.409276 0.209187 0.406901 0.703423 0.713318 0.116874 0.869131 0.189999 0.769990 0.338356 0.104279 0.192549 0.566169 0.850615 0.452414 0.152289 0.031334 0.750211 0.059429 0.060801 0.669301 0.071437 0.102319 0.416501 0.406154 0.020116 0.603459 0.140248 0.550066 0.851914 0.071469 0.033049 0.041489 0.451480 0.957088 0.020321 0.643699 0.112603 0.679204 0.384484 0.040461 0.156702 0.175627 0.490166 0.542379 0.114941 0.917413 0.069953 0.598077 0.091397 0.063959 0.259694 0.451980 0.762526 0.402597 0.020690 0.783838 0.230003 0.669998 0.272284 0.160752 0.949203 0.525915 0.742878 0.284261 0.210239 0.925403 0.473195 0.185094 0.034155 0.333871 0.369996 0.469997 0.021521 0.719862 0.021581 0.163259 0.059532 0.628420 0.354325 0.630827 0.473149 0.036523 0.844987 0.930299 0.632396 0.945953 0.175312 0.514512 0.022291 0.896315 0.671359 0.382884 0.589997 0.849992 0.760074 0.519671 0.103983 0.578920 0.147625 0.373492 0.242660 0.801455 0.150228 0.430450 0.269999 0.909995 0.172165 0.274130 0.305174 0.326353 0.696841 0.181076 0.017377 0.969039 0.023755 0.856526 0.293412 0.594015 0.303830 0.501168 0.226433 0.030063 0.130345 0.728934 0.017623 0.481586 0.291008 0.652482 0.234884 0.970957 0.069949 0.938062 0.360725 0.963258 0.020154 0.402667 0.064390 0.978418 0.099046 0.456622 0.061313 0.548417 0.447419 0.735741 0.756193 0.629484 0.843234 0.336476 0.029986 0.989987 0.800321 0.813299 0.948095 0.666176 0.293063 0.848309 0.070234 0.839275 0.335960 0.185646 0.585192 0.546091 0.170001 0.649991 0.050844 0.495401 0.516366 0.469591 0.292112 0.712525 0.119877 0.821537 0.670253 0.589472 0.020580 0.819557 0.533276 0.128748 0.103180 0.531209 0.305241 0.033240 0.138606 0.089471 0.433958 0.801648 0.541052 0.950881 0.499941 0.399796 0.060649 0.413895 0.388967 0.313406 0.420930 0.599427 0.397513 0.148162 0.329265 0.262507 0.026344 0.077311 0.402046 0.228014 0.029995 0.569994 0.740518 0.942561 0.132388 0.231285 0.614880 0.750538 0.172005 0.828790 0.093296 0.022190 0.056671 0.757886 0.532974 0.312319 0.948957 0.060891 0.033755 0.291631 0.301343 0.778541 0.558187 0.234966 0.155297 0.942129 0.396171 0.886840 0.191752 0.463963 0.516625 0.696879 0.069987 0.149991 0.019991 0.749992 0.072824 0.232388 0.229228 0.611513 0.230249 0.349877 0.843510 0.706575 0.110756 0.631310 0.359019 0.378809 0.352753 0.691935 0.021447 0.120697 0.132228 0.487912 0.230529 0.507897 0.520692 0.773323 0.057045 0.457043 0.346788 0.558919 0.633580 0.285777 0.362315 0.820716 0.262432 0.090839 0.081186 0.498401 0.125282 0.135330 0.170823 0.329609 0.029999 0.009999 0.095802 0.751213 0.070433 0.713054 0.190408 0.090845 0.331119 0.898279 0.030346 0.200606 0.112719 0.973865 0.696591 0.835909 0.210004 0.149995 0.856775 0.581573 0.615396 0.154875 0.282841 0.391298 0.668602 0.485873 0.948112 0.931920 0.367260 0.747130 0.592430 0.447296 0.269997 0.229995 6.337625"
points, score = lecture_vers_donnees(s2)
affichage2(points, attracteurs, [0, 1, 0, 1], 0.01)
print(score)
plt.show()
"""