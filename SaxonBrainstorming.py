import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as Patch
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

ID_1 = totmerge['ID']
x_1 = totmerge['X']
y_1 = totmerge['Y']
z_1 = totmerge['Z']
ID_2 = totmerge2['ID']
x_2 = totmerge2['X']
y_2 = totmerge2['Y']
z_2 = totmerge2['Z']

r_c = totmerge2['r_c']
sig_v = totmerge2['sig_v']
ID_h = totmerge2['ID']
Age_v = totmerge2['Age_x']

ID=harris_p1['ID']
R_gc=harris_p1['R_gc']
Z=harris_p1['Z']
X=harris_p1['X']
Y=harris_p1['Y']
r_c_h=harris_p3['r_c']
v_r=harris_p3['v_r']

FeH_x = totmerge2['FeH_x']
Age_x = totmerge2['Age_x']
FeH_y = totmerge2['FeH_y']
Age_y = totmerge2['Age_y']

rho = harris_p3['rho_0']
sig_h = harris_p3['sig_v']
c = harris_p3['c']
r_t = harris_p3['lg_tc']

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

# Made a plot of Galactocentric radius vs absolute height above the plane of the galaxy, as most of the in situ globular clusters are of decently similar distance, the accreted clusters should be pretty apparent #
R = np.sqrt(X**2+Y**2)

conditions = [v_r<100, (v_r>=100) & (v_r<=300), v_r>300]

cond_colours = ['purple', 'red', 'green']

point_colours = np.select(conditions, cond_colours, default = 'gray')


plt.scatter(R, abs(Z), c=point_colours, alpha=0.3)
plt.xlabel("Galactocentric Radius ($R = \sqrt{X^{2}+Y^{2}}$) (kpc)")
plt.ylabel("Height from Plane |Z| (kpc)")
plt.title("Galactocentric Radius vs Height over Galactic Plane")
plt.plot([10, 10], [0, 110], linestyle='--', color='r', label="10kpc Certainty Range")
plt.ylim(top=110, bottom=0.1)
plt.xlim(left=0.1,right=100)
plt.text(25,70,'# of Possibly Accreted Clusters = 61')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()

conditions2 = [(X**2 + Y**2)**0.5 > 10, (X**2 + Y**2)**0.5 <= 10]

cond_colours2 = ['blue', 'green']

point_colours2 = np.select(conditions2, cond_colours2, default = 'gray')

plt.scatter(X,Y,c=point_colours2, alpha=0.3)
theta = np.linspace(0, 2*np.pi, 500)
r = 10
x = r * np.cos(theta)
y = r * np.sin(theta)
plt.plot(x, y, '--', c='red', lw=1.5, label='10kpc Certainty Range')
plt.axis('equal')
plt.xlabel('Galactic Coordinate X (kpc)')
plt.ylabel('Galactic Coordinate Y (kpc)')
plt.title('Galactic Coordinates X vs Y')
plt.text(-95,22,'# of Possibly Accreted Clusters = 61')
plt.legend()
plt.grid(True, alpha=0.6)
plt.show()


conditions3 = [(x_2**2 + y_2**2)**0.5 > 9.8, (x_2**2 + y_2**2)**0.5 <= 9.8]

cond_colours3 = ['blue', 'green']

point_colours3 = np.select(conditions3, cond_colours3, default = 'gray')

plt.scatter(FeH_x,Age_x,c=point_colours3,alpha=0.3)
for i, txt in enumerate(ID_h):
    if (x_2[i]**2 + y_2[i]**2)**0.5 > 9.8: 
        plt.annotate(txt, (FeH_x[i], Age_x[i]), fontsize=8)
plt.xlabel('[Fe/H]')
plt.ylabel('Age (Gyr)')
plt.title('Age vs Metallicity Plot of Krause21 Clusters')
plt.text(-2.35,8.5,'# of Possibly Accreted Clusters = 19')
plt.legend()
plt.grid(True, alpha=0.6)
plt.show()


plt.scatter(FeH_y,Age_y,c=point_colours3,alpha=0.3)
for i, txt in enumerate(ID_h):
    if (x_2[i]**2 + y_2[i]**2)**0.5 > 9.8: 
        plt.annotate(txt, (FeH_y[i], Age_y[i]), fontsize=8)
plt.xlabel('[Fe/H]')
plt.ylabel('Age (Gyr)')
plt.title('Age vs Metallicity Plot of vandenBerg Clusters')
plt.text(-2.35,9.5,'# of Possibly Accreted Clusters = 12')
plt.legend()
plt.grid(True, alpha=0.6)
plt.show()

R_2 = np.sqrt(x_1**2 + y_1**2)

conditions4 = [R_2 > 10, R_2 <= 10, (R_2 > 10) & (FeH_x >= -1.75)]

cond_colours4 = ['blue', 'green', 'orange']

point_colours4 = np.select(conditions4, cond_colours4, default = 'gray')

plt.scatter(x_1,y_1,c=point_colours4, alpha=0.3)
theta = np.linspace(0, 2*np.pi, 500)
r = 10
x = r * np.cos(theta)
y = r * np.sin(theta)
plt.plot(x, y, '--', c='red', lw=1.5, label='10kpc Certainty Range')
plt.axis('equal')
plt.xlabel('Galactic Coordinate X (kpc)')
plt.ylabel('Galactic Coordinate Y (kpc)')
plt.title('Galactic Coordinates X vs Y')
plt.text(-95,22,'# of Possibly Accreted Clusters = 61')
plt.grid(True, alpha=0.6)
plt.show()