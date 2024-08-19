#Plots a historgram to see the distribution of ships by their time diffrence to their respective images
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

folder = Path('datasets')

files = [file for file in folder.glob('**/2020*.csv')]

dfs = [pd.read_csv(file) for file in files]

bigdf = pd.concat(dfs)

bigdf_water = bigdf[bigdf['waterbody percentage'] == 100]

# bigdf_water['timediff'].plot.hist(bins = 20)
plt.hist(bigdf_water['timediff'], bins = np.arange(0, max(bigdf_water['timediff']), 500))
plt.title('Bins by interval 500')
plt.savefig('500.jpg')

