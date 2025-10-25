"""
Documentation here

"""

# Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

# %% Classification method 1 (Billy)

# Read Krause and vandenBerg csv files
krause = pd.read_csv('Krause21.csv')
vandenberg = pd.read_csv('vandenBerg_table2.csv')

# Rename Krause NGC ID column to 'NGC' in preparation for merging
krause.rename(columns={'Object': '#NGC'}, inplace=True)

# Append 'NGC' to start of each ID in vandenBerg so that it can be merged with krause 
def renameID(ID):
    return 'NGC' + ID

vandenberg['#NGC'] = vandenberg['#NGC'].apply(renameID)

# Start by merging krause and vanden berg datasets
krause_and_vandenberg_merged = pd.merge(krause,vandenberg,on='#NGC')

# Include unique GCs which are missed by merge
krause_and_vandenberg_w_unique = pd.concat([krause_and_vandenberg_merged,krause,vandenberg])
k_and_v_complete = krause_and_vandenberg_w_unique.drop_duplicates(subset=['#NGC'])

# Ensure that all data in FeH and Age columns are numeric or NaN
pd.to_numeric(k_and_v_complete['FeH_x'],errors='coerce')
pd.to_numeric(k_and_v_complete['FeH'],errors='coerce')
pd.to_numeric(k_and_v_complete['Age_x'],errors='coerce')
pd.to_numeric(k_and_v_complete['Age'],errors='coerce')

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
plt.scatter(k_and_v_complete['Age_x'],k_and_v_complete['FeH_x'],c='black',s=7)
plt.ylim(top=-0.5,bottom=-2.5)
plt.xlim(right=15)
plt.title('Metallicity vs. Age of Krause and vandenBerg GCs')
plt.ylabel('[Fe/H]')
plt.xlabel('Age (Gyr)')

plt.figure(1)
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
plt.scatter(k_and_v_results['Age_x'][k_and_v_results['Classification']=='Accreted'],k_and_v_results['FeH_x'][k_and_v_results['Classification']=='Accreted'],c='r',s=15,marker='x',label='Accreted')  
plt.scatter(k_and_v_results['Age_x'][k_and_v_results['Classification']=='In-situ'],k_and_v_results['FeH_x'][k_and_v_results['Classification']=='In-situ'],c='b',s=15,marker='^',label='In-situ')
plt.scatter(k_and_v_results['Age_x'][k_and_v_results['Classification']=='Unsure'],k_and_v_results['FeH_x'][k_and_v_results['Classification']=='Unsure'],c='black',s=7,marker='o',label='Unclassified')
plt.ylim(top=-0.5,bottom=-2.5)
plt.xlim(right=15)
plt.title('Metallicity vs. Age')
plt.ylabel('[Fe/H]')
plt.xlabel('Age (Gyr)')
plt.legend()

# Drop rest of columns so only NGC, Name and classification
classification1 = k_and_v_results.drop(columns = ['Age_x','FeH_x','Age_err','Class','AltName'])

# Show results
plt.show()
print(classification1)

# %% Classification Method 2



# %% Classification Method 3



# %% Merging conditions to determine confidence in classifications

# To merge, make sure all our tables have same column name for ngc/id and
# that all gcs have same naming convention, then do following

# Start by merging our 3 classification results tables

# 2_classifications_merged = pd.merge(classification1,classification2,on='#NGC')

# Then merge this with last classification

# all_classifications_merged = pd.merge(2_classifications_merged,classification3,on='#NGC)


# pd.merge merges tables but only includes things in #NGC column which are column across the
# tables being merged, so to include the unique gcs that were missed by merge do this

# classifications_w_unique = pd.concat([all_classifications_merged,classification1,classification2,classification3])
# classifications_complete = classifications.drop_duplicates(subset=['#NGC'])

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

# classification_values = classifications_complete['Classifcation 1':'Classification 3'].apply(assign_value)

# If this doesn't work try this

# classification_values = classifications_complete['Classifcation 1','Classificatoin 2','Classification 3'].apply(assign_value)

# Now we should have table with ngc column and three columns with either 1, -1 or 0
# Now take the mean across these columns to find the average value which we can use to determine our confidence in
# our classification, as follows

# classification_values['Classification 1':'Classification 3'] = np.mean(classification_values['Classification 1':'Classification 3'],axis=1)

# Again this might not work, if so let me know and I'm happy to debug

# Now we should have table with only 2 columns, one for ngc and one with a float between -1 and 1 i.e. 0.33
# This number represents our percentage confidence that the gc is a certain classification, i.e. 0.33 means
# we are 33% confident that the gc is accreted, and -0.33 means we are 33% sure that the gc is in-situ
# because we assigned in-situ as negative and accreted as positive

# Now present this data however you like, whether it's just the table or if it's in a plot up to you