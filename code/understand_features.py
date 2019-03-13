# import pandas as pd
# import pyarrow.parquet as pq
# import os
# import pyarrow
# import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
# from scipy.signal import *
# import statsmodels.api as sm
# import gc
# from sklearn.feature_selection import f_classif
# import lightgbm as lgbm
# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import expon, uniform, norm
# from scipy.stats import randint, poisson
# from sklearn.metrics import confusion_matrix, make_scorer

# print(os.listdir('../input'))


# print('load data...')
# train_df = pd.read_csv("../input/train_subset.csv")
# print('import ok')


# print(train_df.head())

# plt.figure(figsize=(15, 10))
# plt.title("ID measurement:0, Target:0", fontdict={'fontsize':36})
# plt.plot(train_df["0"].values, marker="o", label='Phase 0')
# plt.plot(train_df["1"].values, marker="o", label='Phase 1')
# plt.plot(train_df["2"].values, marker="o", label='Phase 2')
# plt.ylim(-50,50)
# plt.legend()
# # plt.show()
# plt.savefig('timeseries123.png')


# ts1 = train_df["0"]
# ts2 = train_df["1"]
# ts3 = train_df["2"]

# plt.figure(figsize=(15, 10))
# plt.title("ID measurement:0, Target:0", fontdict={'fontsize':36})
# plt.plot(ts1.rolling(window=100000,center=False).mean(),label='Rolling Mean 1', color='r')
# plt.plot(ts1.rolling(window=100000,center=False).std(),label='Rolling sd 1', color='r')
# plt.plot(ts2.rolling(window=100000,center=False).mean(),label='Rolling Mean 2', color='b')
# plt.plot(ts2.rolling(window=100000,center=False).std(),label='Rolling sd 2', color='b')
# plt.plot(ts3.rolling(window=100000,center=False).mean(),label='Rolling Mean 3', color='g')
# plt.plot(ts3.rolling(window=100000,center=False).std(),label='Rolling sd 3', color='g')
# plt.legend()
# # plt.show()
# plt.savefig('rolling123.png')


# def calc_rolling_amp(row, window=100000):
    # return np.max(row.rolling(window,center=False).mean()) - np.min(row.rolling(window=100000,center=False).mean())

# print(ts1.rolling(window=100000,center=False).mean())

# rolling100k_amp = train_df.apply(calc_rolling_amp)
# print(rolling100k_amp)

# The power spectrum of a time series describes the distribution of power into frequency components composing that signal.

# def welch_max_power_and_frequency(signal):
    # f, Pxx = welch(signal)
    # ix = np.argmax(Pxx)
    # strong_count = np.sum(Pxx>2.5)
    # avg_amp = np.mean(Pxx)
    # sum_amp = np.sum(Pxx)
    # std_amp = np.std(Pxx)
    # median_amp = np.median(Pxx)
    # return [Pxx[ix], f[ix], strong_count, avg_amp, sum_amp, std_amp, median_amp]

# power_spectrum_summary = train_df.apply(welch_max_power_and_frequency, result_type="expand")
# power_spectrum_summary = power_spectrum_summary.T.rename(columns={0:"max_amp", 1:"max_freq", 2:"strong_amp_count", 3:"avg_amp", 
                                                                  # 4:"sum_amp", 5:"std_amp", 6:"median_amp"})
# print(power_spectrum_summary)


# understanding PSD

from scipy import signal
import matplotlib.pyplot as plt

fs = 10e3
N = 1e5
amp = 2*np.sqrt(2)
freq = 1234.0
noise_power = 0.001 * fs / 2
time = np.arange(N) / fs
x = amp*np.sin(2*np.pi*freq*time)
x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)


print(len(x))
print(len(time))

plt.figure()
plt.plot(time, x)
plt.xlabel('time')
plt.ylabel('signal value')
plt.title('2 Vrms sine wave at 1234 Hz, corrupted by \n 0.001 V**2/Hz of white noise sampled at 10 kHz')
# plt.show()
plt.savefig('signal.png')

plt.figure()
f, Pxx_den = signal.welch(x, fs, nperseg=1024)
plt.semilogy(f, Pxx_den)
plt.ylim([0.5e-3, 1])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
# plt.show()
plt.savefig('psd.png')

