import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

harris_p1 = pd.read_csv('HarrisPartI.csv')
harris_p3 = pd.read_csv('HarrisPartIII.csv')
krause = pd.read_csv('Krause21_fixed.csv')
vandenberg = pd.read_csv('vandenBerg_table2_fixed.csv')


# Attempting to merge the Krause21 and Vandenberg data to make plotting easier (there are a few issues with how the tables merge as it removes all unique GC's that are present in Vandenberg but not Krause21) # 
merged = pd.merge(krause, vandenberg, on='ID', how='outer')
print(merged)

# This is a merged data set of the dynamical data featured in HarrisI & HarrisIII (no merging issues here since both data sets have the same GC's) #
merged2 = pd.merge(harris_p1, harris_p3, on='ID', how='outer')
print(merged2)

# This merged data set contains every Harris GC but also gives the common GC's the chemical/age data from the other data sets (still a bit iffy since the original merged set may have cut out some unique GC's from the Vendenberg data set) #
totmerge = pd.merge(merged2, merged, on='ID', how='outer')
print(totmerge)

# This merged data set contains only the GC's that have both dynamical data and chemical/age data (still ironing out some issues) #
totmerge2 = pd.merge(merged, merged2, on='ID', how='left')
print(totmerge2)

# Defining Merged Variables #
# Also defined the age_x colomn for the colourbar, need to find out how to merge this with age_y to get a better representation #
r_c = totmerge2['r_c']
sig_v = totmerge2['sig_v']
ID_h = totmerge2['ID']
Age_v = totmerge2['Age_x']

# Scatter plot of the cluster core radius vs the velocity distribution #
# Added a colourmap and colour index to the scatterplot #
scatter= plt.scatter(r_c, sig_v, c=Age_v, cmap='coolwarm')

# This adds labels to each of the points on the plot just to clarify what point is what GC (i have noticed that for the GC's with chemical/age data NGC5139 does not appear, will have to determine if its accreted through other means) #
for i, txt in enumerate(ID_h):
    plt.annotate(txt, (r_c[i], sig_v[i]), fontsize=8)
plt.colorbar(scatter, label='Age')
plt.xlabel("Core Radius (arcmin)")
plt.ylabel("Velocity Dispersion (km/s)")
plt.title("Core Radius vs Velocity Distribution")
plt.show()

# Even though NGC5139 does not have chemical/age data, the fact that it is such a major outlier in the dynamical data gives me strong belief that its accreted #
# Added colourbars and stuff to experiment with how to better condense our data

# Made a plot of Galacticentric radius vs absolute height above the plane of the galaxy, as most of the in situ globular clusters are of decently similar distance, the accreted clusters should be pretty apparent #
ID=harris_p1['ID']
R_gc=harris_p1['R_gc']
Z=harris_p1['Z']
plt.scatter(R_gc, abs(Z))
for i, txt in enumerate(ID):
    plt.annotate(txt, (R_gc[i], abs(Z[i])), fontsize=8)
plt.xlabel("Galacticentric Radius (kpc)")
plt.ylabel("Height from Plane (kpc)")
plt.title("Galacticentric Radius vs Height over Galactic Plane")
plt.show()

fig, axes = plt.subplots(2,3)

ID_1 = totmerge['ID']
x_1 = totmerge['X']
y_1 = totmerge['Y']
z_1 = totmerge['Z']
ID_2 = totmerge2['ID']
x_2 = totmerge2['X']
y_2 = totmerge2['Y']
z_2 = totmerge2['Z']

axes[0,0].scatter(x_1, y_1)
axes[0,0].set_title('Galactic Distance X vs Y')

axes[0,1].scatter(x_1, z_1)
axes[0,1].set_title('Galactic Distance X vs Z')

axes[0,2].scatter(y_1, z_1)
axes[0,2].set_title('Galactic Distance Y vs Z')

pl1 = axes[1,0].scatter(x_2, y_2, c=Age_v, cmap='coolwarm')
axes[1,0].set_title('Galactic Distance X vs Y')
plt.colorbar(pl1, ax=axes[1,0], label='Age')

pl2 = axes[1,1].scatter(x_2, z_2, c=Age_v, cmap='coolwarm')
axes[1,1].set_title('Galactic Distance X vs Z')
plt.colorbar(pl2, ax=axes[1,1], label='Age')

pl3 = axes[1,2].scatter(y_2, z_2, c=Age_v, cmap='coolwarm')
axes[1,2].set_title('Galactic Distance Y vs Z')
plt.colorbar(pl3, ax=axes[1,2], label='Age')

plt.tight_layout()
plt.show()