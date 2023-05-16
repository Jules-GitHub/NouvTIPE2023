import numpy as np
import pandas as pd

x_moy = np.zeros(20, dtype=float)
y_moy = np.zeros(20, dtype=float)

for i in range(20):
    df = pd.read_csv("Commerces/export" + str(i+1) + ".csv", sep=";")
    x_moy[i] = np.mean(df["X"])
    y_moy[i] = np.mean(df["Y"])

for i in range(20):
    print("x_moy[" + str(i) + "] = " + str(x_moy[i]))
    print("y_moy[" + str(i) + "] = " + str(y_moy[i]))
    print()

pos = open("positions.csv", "w")
pos.write("ARRO;X;Y\n")

xmin = 645441.4871
xmax = 657710.1732
ymin = 6857578.3011
ymax = 6866885.9029

for i in range(20):
    pos.write(str(i+1) + ";" + str((x_moy[i]-xmin)/(xmax-xmin)) + ";" + str((y_moy[i]-ymin)/(ymax-ymin)) + "\n")
