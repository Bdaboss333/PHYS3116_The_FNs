## FINAL TEST
# Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import matplotlib.colors as clrs

# For my analysis, I want to observe the distribution of the glocbular clusters, classfying them against their metallicites, luminosity, and HB type
# The following will predominately analyse the Vandenberg data set

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


def classify3(FeH, lum, HB):
    condFeH_poor  = FeH < -1
    condFeH_rich  = FeH >= -1
    condLum_high  = lum < -9
    condLum_norm  = (lum >= -9) & (lum < -6.5)
    condLum_faint = lum >= -6.5
    condHB_blue   = HB > 0.5
    condHB_mixed  = (HB >= -0.2) & (HB <= 0.5)
    condHB_red    = HB < -0.2
    if (condFeH_rich & condLum_high & condHB_red) or (condFeH_poor & condLum_high & condHB_red) or (condFeH_rich & condLum_norm & condHB_red) or (condFeH_rich & condLum_high & condHB_mixed):
        return 'In-situ'
    elif (condFeH_poor & condLum_faint & condHB_blue) or (condFeH_rich & condLum_faint & condHB_blue) or (condFeH_poor & condLum_norm & condHB_blue) or (condFeH_poor & condLum_faint & condHB_mixed): 
        return 'Accreted'
    elif (condFeH_poor & condLum_high & condHB_blue) or (condFeH_rich & condLum_faint & condHB_red) or (condFeH_poor & condLum_norm & condHB_mixed) or (condFeH_rich & condLum_norm & condHB_mixed):
        return 'Unsure'
    else:
        return 'Unsure'

# Classifying each Globular Cluster
vandenberg['Classification'] = 'Placeholder'

for i in range(len(ID)):
    vandenberg['Classification'].iloc[i] = classify3(FeH.iloc[i],
                                                  lum.iloc[i],
                                                  HB.iloc[i],
                                                  )
    
# Making a new table which mimics the totmerge file but drops every column that isn't 'ID' or 'Classification' to make the data more readable #
classification3 = vandenberg.drop(columns = ['Name','FeH','Age','Age_err','Method','Figs','Range','HBtype','R_G','M_V','v_e0','log_sigma_0'])

# %% Merging conditions to determine confidence in classifications

# Start by merging our 3 classification results tables
classifications_merged2 = pd.merge(classification2,classification1,on='#NGC', how='left')

# Then merge this with last classification
all_classifications_merged = pd.merge(classifications_merged2,classification3,on='#NGC', how='left')

# Now we have finalised table, which should have a column with ngcs/id, then three columns named
# like Classification 1, Classification 2, etc. for classifactions 1 2 and 3, which are either 'Unsure', 
# 'In-situ' or 'Accreted'

# Then from this, use following function with .apply() to assign either -1 to in-situ, 0 to unsure and 1 to accreted

def assign_value(classification):

    if classification == 'In-situ':
        return -1
    elif classification == 'Accreted':
        return 1
    else:
        return 0
    
# Then do the following to make copy of above table with classifications replaced with value from function
classification_values = all_classifications_merged[['Classification_x', 'Classification_y', 'Classification']].applymap(assign_value)
classification_values.insert(0, '#NGC', all_classifications_merged['#NGC'])

classification_cols = ['Classification_x', 'Classification_y', 'Classification']
classification_values['Classification_mean'] = classification_values[classification_cols].mean(axis=1)

def convert(value):

    chance = (1 + abs(value)) / 2 * 100

    if value >= 0:
        return f'{round(chance,2)}% chance Accreted'
    else:
        return f'{round(chance,2)}% chance In-situ'

classification_perc = classification_values[['Classification_mean']].applymap(convert)
classification_perc.rename(columns={'Classification_mean': 'Classification Probability'}, inplace=True)
classification_perc.insert(0, '#NGC', all_classifications_merged['#NGC'])

# Create 3d plot of locations in space coloured by percentaged chance of accreted or insitu

# Set up colour gradient
norm = clrs.Normalize(vmin=-1, vmax=1)
cmap = plt.get_cmap('bwr')

# Map probabilities to colours
colours = cmap(norm(classification_values['Classification_mean']))


plt.figure()
ax = plt.axes(projection='3d')
# Made data point outlines black to see some of the colourmapped points easier #
ax.scatter(totmerge['X'],totmerge['Y'],totmerge['Z'],c=colours,edgecolors='black')
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
cbar.set_label('Accretion Probability\n(-1 = In-Situ, +1 = Accreted)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# ax.view_init(elev=90,azim=90,roll=0)
plt.show()

pd.set_option('display.max_rows', None)
met=totmerge2['FeH_x']
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
