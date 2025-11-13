# Added matplotlib package to plot and fit data, imported as plt to make using the package code easier #
import matplotlib.pyplot as plt
# Added pandas to allow python to read data from our .csv files and to make compiling our data together much easier #
import pandas as pd
# Added numpy to allow for compilation and manipulation of arrays with our astronomical data which makes it easier to plot relationships #
import numpy as np
import astropy.io


# wanna try find some sort of pattern between the FeH in in-situ vs accreted globular clusters



totmerge2 = pd.read_csv('totmerge2.csv')
harris_p1 = pd.read_csv('HarrisPartI.csv')
harris_p3 = pd.read_csv('HarrisPartIII.csv')
krause = pd.read_csv('Krause21.csv')
vandenberg = pd.read_csv('vandenBerg_table2.csv')
# Gonna try add colour to the 3D plot 

# 3D plot of all GC's with metalicity < -1
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

pd.set_option('display.max_rows', None)
met=totmerge2['FeH_x']
sorted_met=sorted(met)
print(sorted_met)

# Scatter plot of the age vs metallicity from merged cluster data but adding colour to represent distance from galactic centre#
r=totmerge2['R_gc']
cond1=r<20
cond2=r>=20

FeH = totmerge2['FeH_x'][cond1]
Age = totmerge2['Age_x'][cond1]
ID = totmerge2['ID'][cond1]
FeH2 = totmerge2['FeH_x'][cond2]
Age2 = totmerge2['Age_x'][cond2]
ID2 = totmerge2['ID'][cond2]

# Added a colourmap to visualize if the GC's group together depending on r #
plt.scatter(Age, FeH, c = 'steelblue', label='r < 20kpc')
plt.scatter(Age2, FeH2, c = 'lightcoral', label='r >= 20kpc')

# plot with titles #
plt.xlabel("Age of merged Clusters (Gyr)")
plt.ylabel("Metallicity of merged Clusters (FeH)")
plt.title("Age vs Metallicity of the merged Globular Clusters w/ colour-coded radial distance")
plt.legend()
plt.show()

# Trying another plot 'Velocity dispersion VS Metallicity'
FeH = totmerge2['FeH_x']
ID = totmerge2['ID']
sigma_v=totmerge2['sig_v']

plt.scatter(FeH, sigma_v, c='goldenrod')

for i, txt in enumerate(ID):
    plt.annotate(txt, (FeH[i], sigma_v[i]), fontsize=8)

plt.xlabel("Metallicity")
plt.ylabel("Velocity dispersion")
plt.title("Velocity dispersion VS Metallicity")
plt.show()

# Code from ChatGPT just to see if anything changes between my code and its code #
# boolean masks
r = pd.to_numeric(totmerge2['R_gc'], errors='coerce')
mask_in  = r < 10
mask_out = r >= 10

# filtered tables (drop missing values if any)
cols = ['ID', 'Age_x', 'FeH_x']
df_in  = totmerge2.loc[mask_in,  cols].dropna()
df_out = totmerge2.loc[mask_out, cols].dropna()

# arrays for plotting/annotating
ID1, Age1, FeH1 = df_in['ID'].to_numpy(),  df_in['Age_x'].to_numpy(),  df_in['FeH_x'].to_numpy()
ID2, Age2, FeH2 = df_out['ID'].to_numpy(), df_out['Age_x'].to_numpy(), df_out['FeH_x'].to_numpy()

# scatter
plt.scatter(Age1, FeH1, color='#0072B2', label='R_gc < 10 kpc')
plt.scatter(Age2, FeH2, color='#D55E00', label='R_gc ≥ 10 kpc')

# annotations (zip avoids index/key errors)
for name, x, y in zip(ID1, Age1, FeH1):
    plt.annotate(name, (x, y), fontsize=8)
for name, x, y in zip(ID2, Age2, FeH2):
    plt.annotate(name, (x, y), fontsize=8)

plt.xlabel('Age (Gyr)')
plt.ylabel('[Fe/H]')
plt.legend()
plt.tight_layout()
plt.show()

# Note: Did not change anything by the looks of it
