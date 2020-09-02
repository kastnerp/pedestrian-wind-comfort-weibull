import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

np.seterr(divide='ignore', invalid='ignore')

df_ws = pd.read_csv("ws.csv", header=None)
df_wdir = pd.read_csv("wdir.csv", header=None)

data = pd.concat([df_ws, df_wdir], axis=1)
data.columns = ['ws', 'wdir']

ws = data['ws'].to_numpy().flatten()
wdir = data['wdir'].to_numpy().flatten()

# a_in = 1
# loc_in = 0
# a_out, Kappa_out, loc_out, Lambda_out = stats.exponweib.fit(data, f0=a_in,floc=loc_in)
a_out, Kappa_out, loc_out, Lambda_out = stats.exponweib.fit(data['ws'].to_numpy())

print("a_out:", a_out)
print("Kappa_out:", Kappa_out)
print("loc_out:", loc_out)
print("Lambda_out:", Lambda_out)

np.mean(data['ws'])
np.median(data['ws'])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

bins_hist = np.linspace(0, 20, 21)
bins_cont = np.linspace(0, 20, 81)

ax.hist(data['ws'], bins=bins_hist, density=True, stacked=True, alpha=0.5)

yy = stats.exponweib.pdf(bins_cont, a=a_out, c=Kappa_out, loc=loc_out, scale=Lambda_out)
xx = bins_cont
ax.plot(xx, yy)

ax.annotate("Shape: $k = %.2f$ \n Scale: $\lambda = %.2f$" % (Kappa_out, Lambda_out), xy=(0.7, 0.85), xycoords=ax.transAxes)
ax.set_ylabel("Probability")
ax.set_xlabel("Wind velocity [m/s]")
plt.show()

# file:///C:/Users/pkastner/Desktop/sustainability-10-00274.pdf


# Exceedance probability
# https://stackoverflow.com/questions/49244352/exceedance-1-cdf-plot-using-seaborn-and-pandas

import numpy as np

ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
sorted_vals = np.sort(ws, axis=None)
exceedance = 1. - np.arange(1., len(sorted_vals) + 1.) / len(sorted_vals)
ax2.plot(sorted_vals, exceedance)


def prob_exceedance(a_theta, u, k, c):
    # not in percentage
    # c = lambda
    return a_theta * math.exp(-(u / c) ** k)


def a_theta(wind_dir, arr):
    return len(np.where((arr < wind_dir + 15) | (arr > wind_dir - 15 + 360))[0]) / len(arr)


pe = prob_exceedance(1, 1, Kappa_out, Lambda_out)
print(pe)

ex_manual = []
for i in bins_cont:
    ex_manual.append(prob_exceedance(1, i, Kappa_out, Lambda_out))

ax2.plot(bins_cont, ex_manual, 'r')

# https://stackoverflow.com/questions/17481672/fitting-a-weibull-distribution-using-scipy
#
# from scipy import stats
# import matplotlib.pyplot as plt
#
# #input for pseudo data
# N = 10000
# Kappa_in = 1.8
# Lambda_in = 10
# a_in = 1
# loc_in = 0
#
# #Generate data from given input
# data = stats.exponweib.rvs(a=a_in,c=Kappa_in, loc=loc_in, scale=Lambda_in, size = N)
#
# #The a and loc are fixed in the fit since it is standard to assume they are known
# a_out, Kappa_out, loc_out, Lambda_out = stats.exponweib.fit(data, f0=a_in,floc=loc_in)
#
# #Plot
# bins = range(51)
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(bins, stats.exponweib.pdf(bins, a=a_out,c=Kappa_out,loc=loc_out,scale = Lambda_out))
# ax.hist(data, bins = bins , density=True, alpha=0.5)
# ax.annotate("Shape: $k = %.2f$ \n Scale: $\lambda = %.2f$"%(Kappa_out,Lambda_out), xy=(0.7, 0.85), xycoords=ax.transAxes)
# plt.show()



# attempt with weib min

# c : array_like

# shape parameters

# loc : array_like, optional

# location parameter (default=0)

# scale : array_like, optional

# scale parameter (default=1)

#https://stackoverflow.com/questions/17481672/fitting-a-weibull-distribution-using-scipy

from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


#Kappa_in = 1.8 # shape
#Lambda_in = 10 # scale


np.seterr(divide='ignore', invalid='ignore')

df_ws = pd.read_csv("ws.csv", header=None)
df_wdir = pd.read_csv("wdir.csv", header=None)

data = pd.concat([df_ws, df_wdir], axis=1)
data.columns = ['ws', 'wdir']

#The a and loc are fixed in the fit since it is standard to assume they are known
Kappa_out, loc_out,  Lambda_out = stats.weibull_min.fit(data['ws'])

#Plot

bins_hist = np.linspace(0, 20, 21)
bins_cont = np.linspace(0, 20, 81)


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(bins_cont, stats.weibull_min.pdf(bins_cont,c=Kappa_out,loc=loc_out,scale = Lambda_out))
ax.hist(data['ws'], bins = bins_hist , density=True, alpha=0.5)
ax.annotate("Shape: $k = %.2f$ \n Scale: $\lambda = %.2f$"%(Kappa_out,Lambda_out), xy=(0.7, 0.85), xycoords=ax.transAxes)

def prob_exceedance(a_theta, u, k, c):
    # not in percentage
    # c = lambda
    return a_theta * math.exp(-(u / c) ** k)


def a_theta(wind_dir, arr):
    return len(np.where((arr < wind_dir + 15) | (arr > wind_dir - 15 + 360))[0]) / len(arr)


pe = prob_exceedance(1, 1, Kappa_out, Lambda_out)
print(pe)

ex_manual = []
for i in bins_cont:
    ex_manual.append(prob_exceedance(1, i, Kappa_out, Lambda_out))

ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(bins_cont, ex_manual, 'r')
ax.set_ylabel("Density")
ax2.set_ylabel("Probability")
ax.set_xlabel("Wind velocity [m/s]")
plt.show()