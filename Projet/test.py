import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.optimize as optimize
import numpy as np
import random as rd
from tqdm import tqdm
import subprocess as sp
import timeit as ti
import os

zone = [0, 4, 0, 4]

def distance(points, point):
    return min([((x - point[0]) ** 2 + (y - point[1]) ** 2) ** 0.5 for x, y in points])

points = np.array([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (0, 4), (4, 0), (2, 0), (2, 4), (0, 2), (4, 2)])
def f(x, y):
    return distance(points, (x, y))

#integrate.dblquad(f, 0, 4, lambda _: 0, lambda _: 4)[0]/16

def affichage(points, e):
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
    plt.show()

#affichage(points, 0.04)

#On utilise ici la méthode Simpson 1/3
def integreD(f, n):
    h0 = (zone[1] - zone[0]) / n
    h1 = (zone[3] - zone[2]) / n
    res = 0
    x = np.arange(zone[0], zone[1], h0/2)
    y = np.arange(zone[2], zone[3], h1/2)
    fxy = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            fxy[i][j] = f(x[i], y[j])
    I = np.zeros(len(x))
    for i in range(len(x)):
        I[i] += h1/6 * (fxy[i][0] + fxy[i][-1] + 4*fxy[i][-2])
        for j in range(1, (len(y)-1)//2):
            I[i] += h1/6 * (4*fxy[i][2*j-1] + 2*fxy[i][2*j])
    res += h0/6 * (I[0] + I[-1] + 4*I[-2])
    for i in range(1, (len(x)-1)//2):
        res += h0/6 * (4*I[2*i-1] + 2*I[2*i])
    return res
    
def test(points):
    def f(x, y):
        return distance(points, (x, y))
    return integreD(f, 100)/((zone[1] - zone[0]) * (zone[3] - zone[2]))

def dist_moyenne(points):
    def f(x, y):
        return distance(points, (x, y))
    return integrate.dblquad(f, zone[0], zone[1], lambda _: zone[2], lambda _: zone[3])[0]/((zone[1] - zone[0]) * (zone[3] - zone[2]))

def minimise(n, e):
    points = np.zeros((n, 2))
    for i in range(1000):
        def f(x, y):
            return distance(points, (x, y))
        maxi = -np.inf
        xMax = -1
        yMax = -1
        for x in range(int(zone[0]/e), int(zone[1]/e)):
            for y in range(int(zone[2]/e), int(zone[3]/e)):
                if f(x*e, y*e) > maxi:
                    maxi = f(x*e, y*e)
                    xMax = x
                    yMax = y
        points2 = np.append(points, [[xMax*e, yMax*e]], axis=0)
        def dist_mut(points, i):
            return distance(np.append(points[:i],points[i+1:], axis=0), points[i])
        inutile = -1
        dist_inutile = np.inf
        for i in range(n):
            if dist_mut(points2, i) < dist_inutile:
                dist_inutile = dist_mut(points2, i)
                inutile = i
        if 0 <= inutile < n:
            points[inutile] = [(points[inutile][0]+xMax*e)/2, (points[inutile][1]+yMax*e)/2]
        else:
            break
    return points

points = minimise([0, 4, 0, 4], 11, 0.04)

affichage(points, [0, 4, 0, 4], 0.04)

dist_moyenne(points)

def dist(pointA, pointB):
    return ((pointA[0] - pointB[0]) ** 2 + (pointA[1] - pointB[1]) ** 2) ** 0.5

def minimise2(n, a1, a2):
    points = []
    for i in range(n):
        points.append([rd.uniform(zone[0], zone[1]), rd.uniform(zone[2], zone[3])])
    vitesses = [[rd.uniform(-0.1, 0.1), rd.uniform(-0.1, 0.1)] for i in range(n)]
    for i in range(1000):
        for j in range(n):
            acc = [0, 0]
            for k in range(n):
                if k != j:
                    acc[0] += (a1/(dist(points[j], points[k])**12) - a2/(dist(points[j], points[k])**6))*(points[j][0] - points[k][0])
                    acc[1] += a1/(dist(points[j], points[k])**12) - a2/(dist(points[j], points[k])**6)*(points[j][1] - points[k][1])
            vitesses[j][0] += acc[0]
            vitesses[j][1] += acc[1]
            points[j][0] += vitesses[j][0]
            points[j][1] += vitesses[j][1]
    return points

points2 = minimise2([0, 4, 0, 4], 11, 0.04, 1, 1)

affichage(points2, [0, 4, 0, 4], 0.04)

def delinearise(x):
    n = len(x)//2
    return np.array([[x[2*i], x[2*i+1]] for i in range(n)])

def linearise(x):
    n = len(x)
    points = np.zeros(2*n)
    for i in range(n):
        points[2*i] = x[i][0]
        points[2*i+1] = x[i][1]
    return points

#Fonction à minimiser de R^2n dans R
def objectif(x):
    n = len(x)//2
    points = [[x[2*i], x[2*i+1]] for i in range(n)]
    return dist_moyenne(points)

points0 = [rd.uniform(0, 4) for i in range(2*5)]
res = optimize.minimize(objectif, points0, method="powell", options={'disp':True})
print(res)
print(points0)

#Méthode de descente du gradient

def gradient(f, x, h=1e-5):
    grad = np.zeros_like(x)
    y = f(x)
    x1 = x.copy()
    for i in range(len(x)):
        x1[i] += h
        grad[i] = (f(x1) - y)/h
        x1[i] -= h
    return grad

def descente(f, x0, h=1e-5, alpha=0.1, epsilon=1e-5, max_iter=1000):
    x = x0
    for i in tqdm(range(max_iter)):
        grad = gradient(f, x, h)
        x = x - alpha*grad
        if np.linalg.norm(grad) < epsilon:
            break
    return x

points0 = [rd.uniform(0, 4) for i in range(2*5)]
gradient(objectif, points0)
res = descente(objectif, points0, max_iter=1000)
affichage(delinearise(res), 0.04)
print(objectif(res))
print(res)

points = delinearise(np.array([
    0.166522, 0.570856, 
    3.137917, 1.919884, 
    2.148045, 0.948821, 
    0.401893, 2.962320,
    2.738161, 3.545057,
    1.567250, 2.157959,
    3.387817, 0.302876,
    0.314303, 2.025476,
    2.660944, 0.106278,
    1.459603, 0.354798
]))
affichage(points, 0.04)
print(dist_moyenne(points))

pointsA = delinearise(np.array([
    0.042857, 0.537378,
    3.054788, 1.858152,
    2.134630, 0.891884,
    0.241409, 2.778063,
    2.719832, 3.681078,
    1.539021, 2.099796,
    3.340220, 0.179173,
    0.238256, 2.094708,
    2.679474, 0.048568,
    1.514031, 0.289017
]))
affichage(pointsA, 0.04)
print(dist_moyenne(pointsA))

pointsB = delinearise(np.array([
    2.987017, 0.397760,
    3.882563, 0.290809,
    0.014273, 2.132319,
    3.418179, 0.313490,
    0.300112, 1.987989,
    0.702985, 1.786824,
    0.446656, 0.183017,
    1.040729, 0.485110,
    1.349067, 3.323387,
    2.475171, 1.134739,
    3.851744, 3.245017,
    2.857297, 2.088265,
    3.315145, 1.119005,
    1.979854, 2.013310,
    0.768735, 0.135791,
    1.084821, 0.017824,
    2.793886, 0.185342,
    2.254999, 3.687159,
    3.345523, 3.538462,
    1.857659, 3.069280
]))
affichage(pointsB, 0.04)
print(dist_moyenne(pointsB))

pointsC = delinearise(np.array([
    3.103636, 3.270739,
    3.806817, 1.355275,
    3.036854, 2.984030,
    2.547021, 0.400060,
    2.617367, 3.212182,
    0.911801, 0.101684,
    0.275913, 3.257035,
    0.805123, 3.198484,
    2.132229, 2.429561,
    1.481170, 2.079321,
    2.704303, 1.531376,
    1.289998, 1.853963,
    2.563899, 3.845326,
    1.609248, 1.988096,
    2.140824, 2.522959,
    2.458459, 3.099013,
    3.577258, 1.384192,
    0.331234, 2.073082,
    0.132388, 3.078854,
    0.461182, 1.112642,
    2.338513, 0.895836,
    0.818795, 3.633852,
    2.806347, 3.228801,
    1.333589, 1.439237,
    0.553035, 1.367758,
    2.429673, 3.794095,
    1.714158, 0.323903,
    2.091581, 1.262796,
    1.393752, 2.562949,
    3.506881, 3.450466,
    3.573924, 1.138390,
    3.164715, 0.715064,
    0.365191, 2.414198,
    3.762497, 1.922423,
    2.194474, 1.579474,
    2.893304, 0.295593,
    2.072940, 2.758098,
    0.208261, 2.286291,
    1.327785, 0.072258,
    2.647295, 1.473611,
    2.614682, 3.527735,
    2.915049, 3.147656,
    3.716591, 2.277805,
    0.414734, 0.766428,
    2.759891, 3.398700,
    2.948572, 3.723584,
    0.210464, 0.227362,
    1.350454, 0.361017,
    2.375014, 3.719612,
    3.772152, 1.535240
]))
affichage(pointsC, 0.04)
print(dist_moyenne(pointsC))

pointsD = delinearise(np.array([
    3.461223, 0.627037,
    2.374719, 0.804934,
    1.219145, 0.513812,
    0.488041, 2.920290,
    1.369287, 3.463134,
    1.716109, 2.046175,
    3.273179, 1.964538,
    0.500454, 1.272013,
    3.511217, 3.304544,
    2.477970, 3.251771
]))
affichage(pointsD, 0.04)
print(dist_moyenne(pointsD))

pointsE = delinearise(np.array([
    1.000076, 2.008466,
    3.000034, 1.991523
]))
affichage(pointsE, 0.04)
print(dist_moyenne(pointsE))

pointsF = delinearise(np.array([
    3.165586, 1.988831,
    1.245829, 3.069505,
    1.231264, 0.943528
]))
affichage(pointsF, 0.04)
print(dist_moyenne(pointsF))

pointsG = delinearise(np.array([
    3.324939, 0.967040,
    0.973553, 3.043560,
    3.013327, 3.046194,
    0.675741, 0.964561,
    1.999811, 1.230366
]))
affichage(pointsG, 0.04)
print(dist_moyenne(pointsG))

pointsH = delinearise(np.array([
    3.291406, 1.474816,
    0.649498, 2.033577,
    0.646523, 0.673664,
    3.274303, 2.500278,
    1.954212, 0.647133,
    1.946300, 1.953040,
    3.322807, 3.507548,
    0.647772, 3.354424,
    3.310979, 0.481449,
    1.955990, 3.315149
]))
affichage(pointsH, 0.04)
print(dist_moyenne(pointsH))

pointsI = delinearise(np.array([
    2.012992, 3.524329,
    3.339227, 2.043687,
    0.649062, 1.968421,
    2.015589, 1.475467,
    3.342358, 3.348664,
    1.958518, 2.522773,
    2.008498, 0.473540,
    0.660405, 0.649109,
    3.350289, 0.686744,
    0.654871, 3.328873
]))
affichage(pointsI, 0.04)
print(dist_moyenne(pointsI))

pointsJ = delinearise(np.array([
    2.000000, 0.646531,
    0.673664, 0.646523,
    0.673345, 1.949619,
    2.012990, 1.938107,
    1.475371, 3.285239,
    3.326952, 0.673048,
    3.340977, 2.004638,
    2.498392, 3.282202,
    0.480614, 3.307006,
    3.505314, 3.341496
]))
affichage(pointsJ, 0.04)
print(dist_moyenne(pointsJ))

pointsK = delinearise(np.array([
    1.946897, 0.622272,
    3.365529, 0.683782,
    2.504310, 1.675984,
    3.482466, 2.270045,
    0.478780, 2.768492,
    3.030941, 3.433302,
    1.192308, 3.562480,
    1.796626, 2.528395,
    0.764962, 1.579314,
    0.636693, 0.513264
]))
affichage(pointsK, 0.04)
print(dist_moyenne(pointsK))

pointsL = delinearise(np.array([
    0.944800, 1.956245,
    0.314921, 3.095286,
    2.111487, 1.305171,
    1.295754, 2.597855,
    1.933194, 0.258787,
    2.665157, 0.926145,
    0.315200, 1.331522,
    3.703957, 0.758581,
    0.239630, 2.500321,
    2.492190, 2.726206,
    3.769201, 1.901372,
    3.641697, 3.756587,
    2.696124, 3.236121,
    2.111257, 3.165143,
    2.119031, 0.755042,
    1.558757, 3.120895,
    1.523916, 2.105296,
    1.920328, 2.552274,
    2.358640, 3.722984,
    0.943039, 3.120394,
    0.749989, 2.519642,
    3.243854, 1.636314,
    3.707764, 0.240743,
    3.135768, 0.356230,
    0.766354, 0.278068,
    3.181099, 1.041407,
    2.985044, 2.114491,
    3.723353, 2.713959,
    3.501445, 2.291992,
    3.729518, 1.318425,
    0.318944, 0.792845,
    0.229060, 3.695431,
    0.321414, 1.904823,
    3.251959, 3.236088,
    1.540351, 0.908152,
    2.471972, 2.200110,
    1.239995, 3.698829,
    0.935892, 1.381908,
    2.658913, 1.575722,
    0.240132, 0.268842,
    1.520329, 1.525333,
    0.936497, 0.826097,
    0.714609, 3.694000,
    1.348802, 0.304182,
    3.064336, 2.696347,
    2.559084, 0.285024,
    3.761514, 3.243288,
    2.973636, 3.747111,
    1.785715, 3.709878,
    2.050822, 1.880625
]))
affichage(pointsL, 0.04)
print(dist_moyenne(pointsL))

pointsM = delinearise(np.array([
    3.346805, 0.673966,
    0.659531, 0.649206,
    1.998139, 3.525547,
    2.011295, 1.475123,
    2.004059, 0.474561,
    0.654273, 1.971785,
    1.987994, 2.523896,
    3.343980, 2.001553,
    3.343272, 3.325279,
    0.653404, 3.326260
]))
affichage(pointsM, 0.04)
print(dist_moyenne(pointsM))

test = sp.run("./a.out 5", capture_output=True)
print(test.stdout.decode("ascii"))

test2 = test.stdout.decode("ascii").split(" ")[:-1]
test3 = list(map(float, test2))
print(test3)

print(ti.timeit(lambda: sp.run("./a.out"), number=5))

X = np.arange(1, 20, 1)

def mini(n):
    sp.run(["gcc", "minimise.c"])
    r = sp.run(["./a.out",str(n)], capture_output=True)
    return float(r.stdout.decode("ascii"))

Y = np.zeros_like(X, dtype=float)
for i in tqdm(range(len(X))):
    Y[i] = mini(X[i])

plt.plot(X, Y)
print(Y)
plt.show()





def calcul_et_affiche(n, e):
    sortie = os.popen("cd Projet && gcc gradient.c && ./a.out "+str(n)).read()
    points = delinearise(list(map(float, sortie.split(" ")[:-1])))
    affichage(points, e)

calcul_et_affiche(20, 0.04)

def minimisation(debut, fin):
    X = np.arange(debut, fin, 1)
    Y = np.zeros_like(X, dtype=float)
    for i in tqdm(range(fin-debut)):
        sortie = os.popen("cd Projet && gcc minimise.c && ./a.out "+str(X[i])).read()
        score = float(sortie)
        Y[i] = score
    plt.plot(X, Y)
    plt.show()

minimisation(1, 21)