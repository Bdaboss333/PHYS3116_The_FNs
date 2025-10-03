# wanna try find some sort of pattern between the FeH in in-situ vs accreted globular clusters

# Defining VanderBerg Variables #
feh = vanderberg['FeH']      # Metallicity
age = vanderberg['Age']      # Cluster Age
ID_v = vanderberg['ID']      # Cluster ID

# Scatter plot of age vs metallicity #
plt.scatter(age, feh)

# Add cluster labels for clarity #
for i, txt in enumerate(ID_v):
    plt.annotate(txt, (age[i], feh[i]), fontsize=6)

plt.xlabel("Age (Gyr)")
plt.ylabel("[Fe/H] (dex)")
plt.title("Age vs Metallicity for Globular Clusters")
plt.show()
