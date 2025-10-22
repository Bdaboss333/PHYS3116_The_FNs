"""
Just taking notes on paper I found

Classification of Accreted and In-Situ stars and cluster

- Using Al/Fe ratio (which I don't think we have)
- Generally, when plotting Age-Metallicity relationship (x=age,y=Fe/H), two tracks form,
  one that goes straight up with age ~ 12.8 Gyr, and one that branches to younger ages
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

# Merging
krause = pd.read_csv('Krause21.csv')
vandenberg = pd.read_csv('vandenBerg_table2.csv')

# Rename Krause NGC ID column to 'NGC' in preparation for merging
krause.rename(columns={'Object': '#NGC'}, inplace=True)

# Append 'NGC' to start of each ID in vanden berg so that it can be merged with krause 
def renameID(ID):
    return 'NGC' + ID

vandenberg['#NGC'] = vandenberg['#NGC'].apply(renameID)

# Start by merging krause and vanden berg datasets
krause_and_vandenberg_merged = pd.merge(krause,vandenberg,on='#NGC')

# Include unique GCs which are missed by merge
krause_and_vandenberg_unique = pd.concat([krause_and_vandenberg_merged,krause,vandenberg])
k_and_v_complete = krause_and_vandenberg_unique.drop_duplicates(subset=['#NGC'])

# Merge age and feh columns of unique gcs
pd.to_numeric(k_and_v_complete['FeH_x'],errors='coerce')
pd.to_numeric(k_and_v_complete['FeH'],errors='coerce')
pd.to_numeric(k_and_v_complete['Age_x'],errors='coerce')
pd.to_numeric(k_and_v_complete['Age'],errors='coerce')
# pd.to_numeric(k_and_v_complete['Age_err'],errors='coerce')


# fillna fn replaces NaN in first column with the chosen data, in this case the second 
# redundant column, 'merging' the two columns
k_and_v_complete['FeH_x'] = k_and_v_complete['FeH_x'].fillna(k_and_v_complete['FeH'])
k_and_v_complete['Age_x'] = k_and_v_complete['Age_x'].fillna(k_and_v_complete['Age'])

# Delete redundant columns
k_and_v_complete.drop(columns = ['FeH','Age'],inplace=True)

# From now on the category I will be looking at is the two branches of the metallicity age relationship
# One branch goes towards younger end while other one goes straight up
# Branch to younger end correlates most often to accreted while straight up is in-situ
# Based on research in 'Accreted versus in situ Milky Way globular clusters' from Duncan A. Forbes, Terry Bridges

# Create bounding box that identifies GCs that we are unsure about
box_xs = np.linspace(12,15,2)
box_ys = np.linspace(-2.6,-1.6,2)

fixed_ys = [-1.6] * len(box_xs)
fixed_xs = [12] * len(box_ys)

# Define a splitting function which separates two branches based on non-linear line of best fit
def split_fn(x):
    y = -0.2 * math.exp(0.3 * x - 1.2) + 1.2
    return y

# Create data points
split_xs = np.linspace(10.4,14.85,50)
split_ys = [split_fn(x) for x in split_xs]


plt.figure(0)
# plt.grid()
plt.scatter(k_and_v_complete['Age_x'],k_and_v_complete['FeH_x'],c='black',s=7)
# plt.errorbar(k_and_v_complete['Age_x'],k_and_v_complete['FeH_x'],xerr=k_and_v_complete['Age_err'],capsize=3,ecolor='black',fmt=" ")
plt.ylim(top=-0.5,bottom=-2.5)
plt.xlim(right=15)
plt.title('Metallicity vs. Age of Krause and vandenBerg GCs')
plt.ylabel('[Fe/H]')
plt.xlabel('Age (Gyr)')

plt.figure(1)
# plt.grid()
plt.plot(split_xs,split_ys,linestyle='-',c='r',label='Classification line')
plt.scatter(k_and_v_complete['Age_x'],k_and_v_complete['FeH_x'],c='black',s=7)
plt.errorbar(k_and_v_complete['Age_x'],k_and_v_complete['FeH_x'],xerr=k_and_v_complete['Age_err'],capsize=3,ecolor='black',fmt=" ")
plt.plot(box_xs,fixed_ys,linestyle='--',c='r',label='Region of uncertainty')
plt.plot(fixed_xs,box_ys,linestyle='--',c='r')
plt.ylim(top=-0.5,bottom=-2.5)
plt.xlim(right=15)
plt.title('Metallicity vs. Age with Error and Classification Regions')
plt.ylabel('[Fe/H]')
plt.xlabel('Age (Gyr)')
plt.legend()

# Determine which GCs are above split line and which below, above being in-situ
# and below being accreted

# Add column to dataset for classification of in-situ or accreted
k_and_v_complete['Classification'] = 'Placeholder'

# Define functions to determine whether in-situ or accreted

# Categorises whether given GC is accreted, in-situ or unsure
def classify(feh,age,age_err):

    # If an age error is unavailable, set it to zero for calculations
    if pd.isna(age_err):
        age_err = np.mean(k_and_v_complete['Age_err'])

    # If GC inside 'unsure box' it is unknown
    if feh <= -1.6 and (age + age_err) >= 12:
        return 'Unsure'

    # If GC error range below split curve it is accreted
    elif feh < -0.2 * math.exp(0.3 * (age + age_err) - 1.2) + 1.2:
        return 'Accreted'
    
    # If GC error range above split curve it is in-situ
    elif feh > -0.2 * math.exp(0.3 * (age - age_err) - 1.2) + 1.2:
        return 'In-situ'
    else:
        return 'Unsure'
    
# Apply function to all GCs in dataset and write output to Classification column
for i in range(len(k_and_v_complete['#NGC'])):
    k_and_v_complete['Classification'].iloc[i] = classify(k_and_v_complete['FeH_x'].iloc[i],
                                                  k_and_v_complete['Age_x'].iloc[i],
                                                  k_and_v_complete['Age_err'].iloc[i])

# Where .iloc ensures that i is interpreted as a position index and isn't searching for an
# FeH that == i, which I mistakenly did before

# Make table with just feh, age, age_err and classification
k_and_v_results = k_and_v_complete.drop(columns = ['Mstar','rh','C5','Name','FeH_y','Age_y','Method','Figs','Range','HBtype','R_G','M_V','v_e0','log_sigma_0'])



# Plot original graph but without errorbars and colour/shape based on whether accreted, in-situ or unsure
plt.figure(2)
# plt.grid()
plt.scatter(k_and_v_results['Age_x'][k_and_v_results['Classification']=='Accreted'],k_and_v_results['FeH_x'][k_and_v_results['Classification']=='Accreted'],c='r',s=15,marker='x',label='Accreted')  
plt.scatter(k_and_v_results['Age_x'][k_and_v_results['Classification']=='In-situ'],k_and_v_results['FeH_x'][k_and_v_results['Classification']=='In-situ'],c='b',s=15,marker='^',label='In-situ')
plt.scatter(k_and_v_results['Age_x'][k_and_v_results['Classification']=='Unsure'],k_and_v_results['FeH_x'][k_and_v_results['Classification']=='Unsure'],c='black',s=7,marker='o',label='Unclassified')
plt.ylim(top=-0.5,bottom=-2.5)
plt.xlim(right=15)
plt.title('Metallicity vs. Age')
plt.ylabel('[Fe/H]')
plt.xlabel('Age (Gyr)')
plt.legend()

plt.show()