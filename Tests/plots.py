import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

import os


df = pd.read_csv("./penguins_fake_real.CSV")
df.drop(["Unnamed: 0"], inplace=True, axis=1)
print(df.head())

#  Plot correlation matrices
# Compute the correlation matrix
def plot_corr_mat(dataframes, outputfile):
    d = dataframes

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    for i in range(0,len(d)):
        corr = d[i].corr()  
    
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        corr_plot = sns.heatmap(corr,cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
    f = corr_plot.get_figure()
    f.savefig(outputfile)

df_fake = df[df['source'] == "fake"].select_dtypes(exclude=['object'])
df_real = df[df['source'] == "real"].select_dtypes(exclude=['object'])

plot_corr_mat(df_fake, "corr_map_fake.png")
plot_corr_mat(df_real, "corr_map_real.png")

# Plot Scree Plots


# Show Loadings


# Pairplots


# Trainloop Plots


# Wasserstein Divergence (im Testloop ?)




