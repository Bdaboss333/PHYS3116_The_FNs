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
totmerge2 = pd.read_csv('totmerge2.csv')
vandenberg_f = pd.read_csv('vandenBerg_table2_fixed.csv')

# Krause21 and vandenBerg_table2 will be primarily used for their age and metallicity
# HarrisPartI and PartIII will be used for their dynamics data

# Testing plots for different data

# 3D plot of all galaxies ts is tuff I love vscode wsg crodie
harris_x = harris_p1['X']
harris_y = harris_p1['Y']
harris_z = harris_p1['Z']
v_r = harris_p3['v_r']
ID_h = harris_p1['ID']


plt.figure(1)
ax = plt.axes(projection='3d')
# Made data point outlines black to see some of the colourmapped points easier #
ax.scatter(harris_x,harris_y,harris_z,c=v_r,edgecolors='black',cmap='RdBu')
# added labels to the plot to make visualisation and determination easier. because there are so many clusters it is important to know which clusters lie where, though for now only the outliers are incredibly important #
for i in range(len(ID_h)):
    ax.text(harris_x[i], harris_y[i], harris_z[i], ID_h [i], color='black', fontsize=8)
# Added a colour bar which maps the heliocentric radial velocities, not sure if it will help too much but its worth a try to see if there is a common trend #
plt.colorbar(ax.collections[0], ax=ax, label='Heliocentric Radial Velocity (km/s)')
ax.set_xlabel('X (kpc)')
ax.set_ylabel('Y (kpc)')
ax.set_zlabel('Z (kpc)')
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
FeH_v = vandenberg_f['FeH']
Age_v = vandenberg_f['Age']
ID_v = vandenberg_f['ID']

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

# Scatter plot of the cluster core radius vs the velocity distribution #
plt.scatter(r_c, sig_v)

# This adds labels to each of the points on the plot just to clarify what point is what GC #
for i, txt in enumerate(ID_h):
    plt.annotate(txt, (r_c[i], sig_v[i]), fontsize=8)
plt.xlabel("Core Radius (arcmin)")
plt.ylabel("Velocity Dispersion (km/s)")
plt.title("Core Radius vs Velocity Distribution")
plt.show()

# Trying out a histrogram plot to see any sort of bimodal relationship
feh = vandenberg['FeH']

plt.hist(feh, bins=15, edgecolor='black')
plt.xlabel("[Fe/H]")
plt.ylabel("Number of Clusters")
plt.title("Bimodal Metallicity Distribution of Globular Clusters")
plt.show()

# Gonna try add colour to the 3D plot to distinguish between accreted and in-situ GC's

# 3D plot of all GC's with metallicity < -1.3 and > -1.3
x_coord = totmerge2['X'][totmerge2['FeH_x']<-1.3]
y_coord = totmerge2['Y'][totmerge2['FeH_x']<-1.3]
z_coord= totmerge2['Z'][totmerge2['FeH_x']<-1.3]
x_coord2 = totmerge2['X'][totmerge2['FeH_x']>-1.3]
y_coord2 = totmerge2['Y'][totmerge2['FeH_x']>-1.3]
z_coord2= totmerge2['Z'][totmerge2['FeH_x']>-1.3]

plt.figure(1)
ax = plt.axes(projection='3d')
ax.plot3D(x_coord,y_coord,z_coord,'o', c='red')
ax.plot3D(x_coord2,y_coord2,z_coord2,'o', c='blue')
ax.view_init(elev=30,azim=0,roll=0)
plt.show()

# Scatter plot of the age vs metallicity from merged cluster data but adding colour to represent distance from galactic centre#
r=np.sqrt(totmerge2['X']**2 + totmerge2['Y']**2 + totmerge2['Z']**2)
cond1=r<10
cond2=r>=10

FeH = totmerge2['FeH_x'][cond1]
Age = totmerge2['Age_x'][cond1]
ID = totmerge2['ID'][cond1]
FeH2 = totmerge2['FeH_x'][cond2]
Age2 = totmerge2['Age_x'][cond2]
ID2 = totmerge2['ID'][cond2]

# Added a colourmap to visualize if the GC's group together depending on r #
plt.scatter(Age, FeH, c = 'steelblue', label='r < 10kpc')
plt.scatter(Age2, FeH2, c = 'lightcoral', label='r >= 10kpc')

# plot with titles #
plt.xlabel("Age of merged Clusters (Gyr)")
plt.ylabel("Metallicity of merged Clusters (FeH)")
plt.title("Age vs Metallicity of the merged Globular Clusters w/ colour-coded radial distance")
plt.legend()
plt.show()

# last test
x_coord = harris_p1['X'][harris_p3['rho_0']<1.5]
y_coord = harris_p1['Y'][harris_p3['rho_0']<1.5]
z_coord= harris_p1['Z'][harris_p3['rho_0']<1.5]
x_coord2 = harris_p1['X'][harris_p3['rho_0']>1.5]
y_coord2 = harris_p1['Y'][harris_p3['rho_0']>1.5]
z_coord2= harris_p1['Z'][harris_p3['rho_0']>1.5]

plt.figure(1)
ax = plt.axes(projection='3d')
ax.plot3D(x_coord,y_coord,z_coord,'o', c='steelblue')
ax.plot3D(x_coord2,y_coord2,z_coord2,'o', c='lightcoral')
ax.view_init(elev=30,azim=0,roll=0)
plt.show()

pd.set_option('display.max_rows', None)
met=harris_p3['rho_0']
sorted_met=sorted(10**met)
print(sorted_met)
