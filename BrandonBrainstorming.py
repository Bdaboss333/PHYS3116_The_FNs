# Added matplotlib package to plot and fit data, imported as plt to make using the package code easier #
import matplotlib.pyplot as plt
# Added pandas to allow python to read data from our .csv files and to make compiling our data together much easier #
import pandas as pd
# Added numpy to allow for compilation and manipulation of arrays with our astronomical data which makes it easier to plot relationships #
import numpy as np
import astropy.io

# wanna try find some sort of pattern between the FeH in in-situ vs accreted globular clusters



totmerge2 = pd.read_csv('totmerge2.csv')
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