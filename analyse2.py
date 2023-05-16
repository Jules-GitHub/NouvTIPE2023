import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

couleurs = ["red", "blue", "green", "yellow", "black", "orange", "purple", "pink", "brown", "grey", "cyan", "magenta", "olive", "lime", "teal", "navy", "coral", "gold", "silver", "maroon"]

xmin = np.inf
xmax = -np.inf
ymin = np.inf
ymax = -np.inf

for i in range(20):
    df = pd.read_csv("Commerces/export" + str(i+1) + ".csv", sep=";")
    for x,y in zip(df["X"], df["Y"]):
        if x < xmin:
            xmin = x
        if x > xmax:
            xmax = x
        if y < ymin:
            ymin = y
        if y > ymax:
            ymax = y
        plt.plot(x, y, "o", color=couleurs[i])

print("xmin = " + str(xmin))
print("xmax = " + str(xmax))
print("ymin = " + str(ymin))
print("ymax = " + str(ymax))

plt.show()