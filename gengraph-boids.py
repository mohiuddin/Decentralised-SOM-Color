import math
from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.stats as st
import pandas as pd


fileNames = ['Flocking', 'Lines', 'Spermatozoa', 'Old Man', 'Ink', 'Gravity', 'Firefly', 'Busy Bees',
             'Random 1', 'Random 2', 'Random 3', 'Random 4', 'Random 5', 'Random 6', 'Random 7','Random 8']

means = np.zeros((5,1,16))   # depth, row, column
confs = np.zeros((5,1,16))
samples = np.zeros((4,1,16))

for i in range(len(fileNames)):
    toread = fileNames[i]+".npz"
    temp = np.load(toread)
    data = temp["name1"]
    sp = temp["name2"]
    print(sp)
    for j in range(5):
        colj = data[:,j]

        tempmean = np.mean(colj)
        ci = st.t.interval(alpha=0.95, df=len(colj) - 1, loc=tempmean, scale=st.sem(colj))
        limit = tempmean - ci[0]
        confs[j, 0, i] = limit
        means[j, 0, i] = tempmean
        #if j <= 4:
            #spj = sp[:, j]
            #samples[j,0,i] = spj

a = list(range(1,17))
b = means[1,:,:]
errors = confs[1,:,:]

b = b[0,:]
errors = errors[0,:]


fig, ax = plt.subplots()
plt.scatter(b,a)
plt.xlim([0,12])
ax.set_yticks(a)
ax.set_yticklabels(fileNames)
plt.errorbar(b,a,xerr=errors, linestyle="None", fmt='o-')
plt.show()
