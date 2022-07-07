import pandas as pd
import seaborn as sns

im_file = pd.read_csv("175-FIX.csv", index_col=False)
sns.set(rc = {'figure.figsize':(16,8)})
sns.heatmap(im_file.corr(), annot = True, fmt='.2g',cmap= 'coolwarm')