import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
r_c = totmerge2['r_c']
sig_v = totmerge2['sig_v']
ID_h = totmerge2['ID']

# Scatter plot of the cluster core radius vs the velocity distribution #
plt.scatter(r_c, sig_v)

# This adds labels to each of the points on the plot just to clarify what point is what GC (i have noticed that for the GC's with chemical/age data NGC5139 does not appear, will have to determine if its accreted through other means) #
for i, txt in enumerate(ID_h):
    plt.annotate(txt, (r_c[i], sig_v[i]), fontsize=8)
plt.xlabel("Core Radius (arcmin)")
plt.ylabel("Velocity Dispersion (km/s)")
plt.title("Core Radius vs Velocity Distribution")
plt.show()

# Even though NGC5139 does not have chemical/age data, the fact that it is such a major outlier in the dynamical data gives me strong belief that its accreted #