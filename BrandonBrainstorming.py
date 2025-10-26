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
met=[-0.58, -0.05, 0.01, 0.03, 0.11, 0.38, 0.54, 0.84, 0.84, 1.1, 1.53, 1.55, 1.62, 1.78, 2.06, 2.57, 2.71, 2.9, 2.99, 3.0, 3.07, 3.51, 3.57, 3.63, 3.63, 3.95, 4.06, 4.74, 4.88, 4.08, 5.09, 3.88, 4.1, 4.66, -0.97, -0.25, 0.97, 1.65, 1.82, 2.31, 2.5, 2.97, 3.08, 3.17, 3.23, 3.41, 3.64, 3.78, 3.78, 3.78, 4.09, 4.23, 4.68, 2.39, 3.15, 3.54, 3.54, 3.55, 4.3, 4.49, 4.55, 4.61, 4.67, 4.79, 3.98, 4.86, 4.97, 5.04, 5.16, 5.31, 5.52, 5.59, 1.99, 2.29, 2.34, 2.47, 2.68, 2.78, 2.85, 3.36, 3.4, 3.46, 3.63, 3.73, 3.79, 3.84, 3.85, 3.85, 4.15, 4.33, 4.41, 4.42, 4.64, 4.65, 4.77, 4.95, 5.14, 5.24, 5.26, 5.28, 5.29, 5.35, 5.37, 5.48, 5.76, 5.83, 5.85, 6.06, -0.25, 0.16, 0.53, 1.27, 1.63, 2.22, 2.34, 2.38, 2.58, 2.79, 3.18, 3.28, 3.3, 3.84, 4.09, 4.86, 2.29, 2.83, 3.33, 3.44, 3.51, 3.63, 3.64, 3.89, 4.0, 4.48, 4.51, 4.58, 4.64, 4.69, 5.01, 5.04, 5.05, 5.3, 5.82]
sorted_met=sorted(met)
print(sorted_met)

# Scatter plot of the age vs metallicity from merged cluster data but adding colour to represent distance from galactic centre#
r=np.sqrt(totmerge2['X']**2 + totmerge2['Y']**2 + totmerge2['Z']**2)
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


# last test
condition1=harris_p3['rho_0']


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








## FINAL TEST
# For my analysis, I want to observe the distribution of the glocbular clusters, classfying them against their metallicites, luminosity, and HB type
# The following will predominately analyse the Vandenberg data set

# ----- Importing packages and defining variables ----- #
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import astropy.io

totmerge2 = pd.read_csv('totmerge2.csv')
harris_p1 = pd.read_csv('HarrisPartI.csv')
harris_p3 = pd.read_csv('HarrisPartIII.csv')
krause = pd.read_csv('Krause21.csv')
vandenberg = pd.read_csv('vandenBerg_table2.csv')


# Defining variables
FeH=vandenberg['FeH']
lum=vandenberg['M_V']
HB=vandenberg['HBtype']
ID=vandenberg['#NGC']

# Defining conditions based on academic papers and resources
# In-situ GC's tend to be metal-rich, redder, and brighter compared to accreted GC's.
# Under this assumption, we can define the following function to classify the Vandenberg clusters into "In-situ", "Accreted", or "unsure"
condFeH_poor  = FeH < -1
condFeH_rich  = FeH >= -1

condLum_high  = lum < -9
condLum_norm  = (lum >= -9) & (lum < -6.5)
condLum_faint = lum >= -6.5

condHB_blue   = HB > 0.5
condHB_mixed  = (HB >= -0.2) & (HB <= 0.5)
condHB_red    = HB < -0.2

def classify(FeH, lum, HB):
    if (condFeH_rich) & (condLum_high) & (condHB_red):
        return 'In-situ'
    elif (condFeH_poor) & (condLum_high) & (condHB_red): 
        return 'In-situ'
    elif (condFeH_rich) & (condLum_norm) & (condHB_red):
        return 'In-situ'
    elif (condFeH_rich) & (condLum_high) & (condHB_mixed):
        return 'In-situ'
    elif (condFeH_poor) & (condLum_faint) & (condHB_blue): 
        return 'Accreted'
    elif (condFeH_rich) & (condLum_faint) & (condHB_blue):
        return 'Accreted'
    elif (condFeH_poor) & (condLum_norm) &  (condHB_blue):
        return 'Accreted'
    elif (condFeH_poor) & (condLum_faint) & (condHB_mixed):
        return 'Accreted'
    elif (condFeH_poor) & (condLum_high) & (condHB_blue):
        return 'Unsure'
    elif (condFeH_rich) & (condLum_faint) & (condHB_red):
        return 'Unsure'
    elif (condFeH_poor) & (condLum_norm) & (condHB_mixed):
        return 'Unsure'
    elif (condFeH_rich) & (condLum_norm) & (condHB_mixed):
        return 'Unsure'

# Classifying each Globular Cluster
vandenberg['Classification'] = 'Placeholder'

for i in range(len('#NGC')):
    FeH_val = vandenberg.loc[i, 'FeH']
    lum_val = vandenberg.loc[i, 'M_V']
    HB_val  = vandenberg.loc[i, 'HBtype']
    vandenberg.loc[i, 'Classification'] = classify(FeH, lum, HB)


# Making a 3D plot colour-coded based off the classfication to observe the distribution of globular clusters in the Milky way 
 


    