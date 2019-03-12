import pandas as pd
import pyarrow.parquet as pq
import os
#import pyarrow
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

print(os.listdir('../input'))


print('load data...')
train = dd.read_parquet('../input/train.parquet')
#train = train.compute()
#rain = pd.read_parquet('../input/train.parquet', engine='fastparquet')
train_df = pq.read_pandas("../input/train.parquet").to_pandas()
train_meta_df = pd.read_csv("../input/metadata_train.csv")
print('import ok')

# print(train.info())
# print(train.head())
# print(train.shape)

# plot settings
rand_seed = 135
np.random.seed(rand_seed)
xsize = 12.0
ysize = 8.0

#fig, ax = plt.subplots()
#ax = sns.countplot(x="phase", hue="target", data=train_meta_df, ax=ax)
#ax.set_title("Number of normal vs anormal observation for each phase")
#plt.show()

print(train["0"].values)

#plt.figure(figsize=(15, 10))
#plt.title("ID measurement:0, Target:0", fontdict={'fontsize':36})
#plt.plot(train["0"].values, marker="o", label='Phase 0')
#plt.plot(train["1"].values, marker="o", label='Phase 1')
#plt.plot(train["2"].values, marker="o", label='Phase 2')
#plt.ylim(-50,50)
#plt.legend()
#plt.show()

