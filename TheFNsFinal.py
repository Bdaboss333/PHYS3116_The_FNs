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

# %% Classification Method 2 (Saxon)

# Read csv files
harris_p1 = pd.read_csv('HarrisPartI.csv')
harris_p3 = pd.read_csv('HarrisPartIII.csv')
krause = pd.read_csv('Krause21_fixed.csv')
vandenberg = pd.read_csv('vandenBerg_table2_fixed.csv')
totmerge = pd.read_csv('totmerge.csv')
totmerge2 = pd.read_csv('totmerge2.csv')

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

# ==== Dynamical Plots of X vs Y Positions and Galactocentric Radius vs |Z| ==== #

# Defining the Galactocentric Radius for the Harris data and finding its standard deviation #
R = np.sqrt(X**2+Y**2)
R_std = np.std(R)

# Placing conditions on the Galactocentric Radius vs |Z| plot to colour code by # of standard deviations away from the mean #
conditions = [R>3*R_std, (R>R_std) & (v_r<3*R_std), R<R_std]
cond_colours = ['purple', 'red', 'green']
point_colours = np.select(conditions, cond_colours, default = 'red')

# Plot of Galactocentric Radius vs |Z| #
plt.scatter(R, abs(Z), c=point_colours, alpha=0.3)
plt.xlabel("Galactocentric Radius ($R = \sqrt{X^{2}+Y^{2}}$) (kpc)")
plt.ylabel("Height from Plane (kpc)")
plt.title("Galactocentric Radius vs Height over Galactic Plane")
plt.plot([10, 10], [0, 110], linestyle='--', color='r', label="10kpc Certainty Range")
plt.ylim(top=110, bottom=0.1)
plt.xlim(left=0.1,right=100)
plt.text(25,70,'# of Possibly Accreted Clusters = 61')
plt.legend()
plt.grid(True)
plt.show()

# Placing conditions on the X vs Y Position plot #
conditions2 = [(X**2 + Y**2)**0.5 > 10, (X**2 + Y**2)**0.5 <= 10]
cond_colours2 = ['blue', 'green']
point_colours2 = np.select(conditions2, cond_colours2, default = 'gray')

# X vs Y position plot #
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

# ==== Age vs Metallicity Plots ==== #

# Placing conditions on the Age vs Metallicity plots to colour the points which lie outside the 10kpc certainty area #
conditions3 = [(x_2**2 + y_2**2)**0.5 > 9.8, (x_2**2 + y_2**2)**0.5 <= 9.8]
cond_colours3 = ['blue', 'green']
point_colours3 = np.select(conditions3, cond_colours3, default = 'gray')

# Age vs Metallicity plot using Krause21 data #
plt.scatter(FeH_x,Age_x,c=point_colours3,alpha=0.3)

# Labelling each point which lies outside of both the 10kpc certainty area and the uncertainty area on the plot #
for i, txt in enumerate(ID_h):
    if ((x_2[i]**2 + y_2[i]**2)**0.5 > 9.8) & (FeH_x[i] > -1.75): 
        plt.annotate(txt, (FeH_x[i], Age_x[i]), fontsize=8)
plt.plot([-1.75, -1.75], [12, 14.5], linestyle='--', color='r', label="Uncertainty Range")
plt.plot([-2.5, -1.75], [12, 12], linestyle='--', color='r')
plt.ylim(top=14.5, bottom = 7)
plt.xlim(left=-2.5, right=0)
plt.xlabel('[Fe/H]')
plt.ylabel('Age (Gyr)')
plt.title('Age vs Metallicity Plot of Krause21 Clusters')
plt.text(-2.35,8.5,'# of Possibly Accreted Clusters = 19')
plt.legend()
plt.grid(True, alpha=0.6)
plt.show()

# Age vs Metallicity plot using the VandenBerg data #
plt.scatter(FeH_y,Age_y,c=point_colours3,alpha=0.3)

# Labelling each point which lies outside of the 10kpc certainty area #
for i, txt in enumerate(ID_h):
    if (x_2[i]**2 + y_2[i]**2)**0.5 > 9.8: 
        plt.annotate(txt, (FeH_y[i], Age_y[i]), fontsize=8)
plt.xlabel('[Fe/H]')
plt.ylabel('Age (Gyr)')
plt.title('Age vs Metallicity Plot of vandenBerg Clusters')
plt.text(-2.35,9.5,'# of Possibly Accreted Clusters = 13')
plt.grid(True, alpha=0.6)
plt.show()

# Defining the Galactocentric Radius for the totmerge data and finding its standard deviation #
R_2 = np.sqrt(x_1**2 + y_1**2)
R_2_std = np.std(R_2)

# Set conditions upon the data to color code based on certainty of accretion #
conditions4 = [(R_2 > 10) & (FeH_x >= -1.75) | (R_2 > 10) & (FeH_y >= -1.75) | (R_2>3*R_2_std), R_2 > 10, R_2 <= 10]
cond_colours4 = ['purple', 'blue', 'green']
point_colours4 = np.select(conditions4, cond_colours4, default = 'gray')

# ==== Final X vs Y Position Plot with Classification Colour Coding ==== #

# X vs Y position plots of the totmerge data with colour coding and a 10kpc certainty area #
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

# Add column to dataset for classification of in-situ or accreted or unsure #
totmerge['Classification'] = 'Placeholder'

# Define functions to determine whether in-situ or accreted #

# Categorises whether given GC is accreted, in-situ or unsure #
def classify(FeH_x,Age_x,FeH_y,R_2):
    if (R_2 > 9.8) & (FeH_x >= -1.75) | (R_2 > 9.8) & (FeH_y >= -1.75) | (R_2>3*R_2_std):
        return 'Accreted'
    elif (R_2 > 9.8) | (FeH_x>=-1.75) & (Age_x<12.5) | (FeH_x<=-1.75) & (Age_x>=12.5):
        return 'Unsure'
    elif (R_2 <= 9.8):
        return 'In-Situ'

# Using an .iloc command to determine classifications using the conditions set above by looking at each indexed row #
for i in range(len(totmerge['ID'])):
    totmerge['Classification'].iloc[i] = classify(totmerge['FeH_x'].iloc[i],
                                                  totmerge['Age_x'].iloc[i],
                                                  totmerge['FeH_y'].iloc[i],
                                                  R_2.iloc[i])

# Making a new table which mimics the totmerge file but drops every column that isn't 'ID' or 'Classification' to make the data more readable #
classification = totmerge.drop(columns = ['Mstar','rh','C5','Name_x','Name_y','FeH_y','Age_y','Method','Figs','Range','HBtype','R_G','M_V','v_e0','log_sigma_0','Age_x','FeH_x','Age_err','AltName','X','Y','Z','RA','DEC','L','B','R_Sun','R_gc','v_r','v_r_e','v_LSR','sig_v','sig_v_e','c','r_c','r_h','mu_V','rho_0','lg_tc','lg_th'])

# Renaming the 'ID' column to '#NGC' to match with Billy's data for easy merging # 
classification2 = classification.rename(columns={'ID': '#NGC'})

# Removes the truncation of the data in terminal so I can read it #
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(classification2)
# %% Classification Method 3



# %% Merging conditions to determine confidence in classifications

# ===== Import packages ==== #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# To merge, make sure all our tables have same column name for ngc/id and
# that all gcs have same naming convention, then do following

# Start by merging our 3 classification results tables

# 2_classifications_merged = pd.merge(classification1,classification2,on='#NGC')

classifications_merged2 = pd.merge(classification1,classification2,on='#NGC')
print(classifications_merged2)

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

# nvm im a bot no it's not here's new function to convert the number into percentage chance

def convert(value):

    if value >= 0:
        chance = (1 + abs(value)) / 2 * 100
    else:
        chance = -((1 + abs(value)) / 2 * 100)

    return f'{chance}%'

# Now present this data however you like, whether it's just the table or if it's in a plot up to you
# %%
