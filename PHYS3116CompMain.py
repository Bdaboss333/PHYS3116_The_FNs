# Added matplotlib package to plot and fit data, imported as plt to make using the package code easier #
import matplotlib.pyplot as plt
# Added pandas to allow python to read data from our .csv files and to make compiling our data together much easier #
import pandas as pd
# Added numpy to allow for compilation and manipulation of arrays with our astronomical data which makes it easier to plot relationships #
import numpy as np
import astropy.io



# Read all data csvs
harris_p1 = pd.read_csv('HarrisPartI.csv')
harris_p3 = pd.read_csv('HarrisPartIII.csv')
krause = pd.read_csv('Krause21.csv')
vandenberg = pd.read_csv('vandenBerg_table2.csv')

# Krause21 and vandenBerg_table2 will be primarily used for their age and metallicity
# HarrisPartI and PartIII will be used for their dynamics data

# Testing plots for different data

# 3D plot of all galaxies ts is tuff I love vscode wsg crodie
harris_x = harris_p1['X']
harris_y = harris_p1['Y']
harris_z = harris_p1['Z']

plt.figure(1)
ax = plt.axes(projection='3d')
ax.plot3D(harris_x,harris_y,harris_z,'o')
ax.view_init(elev=30,azim=0,roll=0)
plt.show()

# Scatter plot of the age vs metallicity from Krause 21 cluster data #
FeH_k = krause['FeH']
Age_k = krause['Age']
ID_k = krause['Object']

# Added a colourmap to visualize how the GC colours would change with age #
plt.scatter(Age_k, FeH_k, c = Age_k, cmap = 'coolwarm')

# This adds labels to each of the points on the plot just to clarify what point is what GC #
for i, txt in enumerate(ID_k):
    plt.annotate(txt, (Age_k[i], FeH_k[i]), fontsize=8)
plt.xlabel("Age of Krause Clusters (Gyr)")
plt.ylabel("Metallicity of Krause Clusters (FeH)")
plt.title("Age vs Metallicity of the Krause Globular Clusters")
plt.show()

# Scatter plot of the age vs metallicity from VanderBerg data #
FeH_v = vandenberg['FeH']
Age_v = vandenberg['Age']
ID_v = vandenberg['Name']

# Added a colourmap to visualize how the GC colours would change with age #
plt.scatter(Age_v, FeH_v, c = Age_v, cmap = 'coolwarm')

# This adds labels to each of the points on the plot just to clarify what point is what GC #
for i, txt in enumerate(ID_v):
    plt.annotate(txt, (Age_v[i], FeH_v[i]), fontsize=8)
plt.xlabel("Age of VanderBerg Clusters (Gyr)")
plt.ylabel("Metallicity of VanderBerg Clusters (FeH)")
plt.title("Age vs Metallicity of the VanderBerg Globular Clusters")
plt.show()

# Defining Harris Variables #
r_c = harris_p3['r_c']
sig_v = harris_p3['sig_v']
ID_h = harris_p1['ID']

# Scatter plot of the cluster core radius vs the velocity distribution #
plt.scatter(r_c, sig_v)

# This adds labels to each of the points on the plot just to clarify what point is what GC #
for i, txt in enumerate(ID_h):
    plt.annotate(txt, (r_c[i], sig_v[i]), fontsize=8)
plt.xlabel("Core Radius (arcmin)")
plt.ylabel("Velocity Dispersion (km/s)")
plt.title("Core Radius vs Velocity Distribution")
plt.show()

