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
import astropy.io

# Attempting to merge
harris_p1 = pd.read_csv('HarrisPartI.csv')
harris_p3 = pd.read_csv('HarrisPartIII.csv')
krause = pd.read_csv('Krause21.csv')
vandenberg = pd.read_csv('vandenBerg_table2.csv')


krause.rename(columns={'Object': '#NGC'}, inplace=True)

def renameID(ID):
    return 'NGC' + ID

vandenberg['#NGC'] = vandenberg['#NGC'].apply(renameID)

# Method 1
krause_and_vandenberg_merged = pd.merge(krause,vandenberg,on='#NGC')

# Method 2 to include unique GCs
# krause_and_vandenberg_merged = pd.concat([krause,vandenberg])
# k_and_v_fixed = krause_and_vandenberg_merged.drop_duplicates(subset=['#NGC'])

# From now on the category I will be looking at is the two branches of the metallicity age relationship
# One branch goes towards younger end while other one goes straight up
# Branch to younger end correlates most often to accreted while straight up is in-situ

# Find an equation that splits two branches

y_split_xs = np.linspace(12,15,8)
x_split_ys = np.linspace(-2.6,-1.6,8)

y_split_ys = [-1.6] * len(y_split_xs)
x_split_xs = [12] * len(x_split_ys)

print(y_split_ys)
print(x_split_xs)

plt.figure(0)
plt.scatter(krause_and_vandenberg_merged['Age_x'],krause_and_vandenberg_merged['FeH_x'],c='red')
plt.errorbar(krause_and_vandenberg_merged['Age_x'],krause_and_vandenberg_merged['FeH_x'],yerr=krause_and_vandenberg_merged['Age_err'],capsize=5,ecolor='red',fmt=" ")
plt.plot(y_split_xs,y_split_ys,linestyle='--',c='black')
plt.plot(x_split_xs,x_split_ys,linestyle='--',c='black')
plt.ylim(bottom=-2.6)
plt.xlim(right=15)
plt.title('Metallicity vs. Age')
plt.ylabel('[Fe/H]')
plt.xlabel('Age (Gyr)')
plt.show()
