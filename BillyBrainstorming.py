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
# krause_and_vandenberg_merged = pd.merge(krause,vandenberg,on='#NGC')

# Method 2 to include unique GCs
krause_and_vandenberg_merged = pd.concat([krause,vandenberg])
k_and_v_fixed = krause_and_vandenberg_merged.drop_duplicates(subset=['#NGC'])

# plt.scatter(krause_and_vandenberg_merged['Age_x'],krause_and_vandenberg_merged['FeH_x'])
# plt.show()


print(k_and_v_fixed)