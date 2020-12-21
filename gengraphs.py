import math
from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.stats as st
import pandas as pd

'''
This script generates star, mesh or ring topology graphs from saved npz files.
Solely proesses data, does not do any processing


'''
#fileNames = ['star2.npz', 'star4.npz', 'star8.npz']
fileNames = ['mesh2.npz', 'mesh4.npz', 'mesh8.npz']
#fileNames = ['ring2.npz', 'ring4.npz', 'ring8.npz']
data = np.zeros((3, 10, 5))
samples = np.zeros((3,4))

for i in range(3):
    temp = np.load(fileNames[i])
    data[i] = temp['name1']
    tt = temp['name2']
    samples[i, :] = np.reshape(tt, 4)
    #print(data[i])

confs = np.zeros((3,5))
means = np.zeros((3,5))
for i in range(3):
    coldata = data[i]
    for j in range (5):
        colj = coldata[:,j]
        tempmean = np.mean(colj)
        ci = st.t.interval(alpha = 0.95, df = len(colj)-1, loc = tempmean, scale = st.sem(colj))
        limit = tempmean - ci[0]
        confs[i,j] =  limit
        means[i,j] = tempmean

print(means)
print(samples)
print(confs)

#Now lets plot
length = len(means)
x_labels = ['Limit = 2', 'Limit = 4', 'Limit = 8']

fig, ax = plt.subplots()
width = 0.1
x = np.arange(length)

ax.bar(x, means[:,0], width, color='#820088', label='Centralised', yerr=confs[:,0])
ax.bar(x + width, means[:,1], width, color='#0F52BA', label='Agent 1', yerr=confs[:,1])
ax.bar(x + (2 * width), means[:,2], width, color='#6593F5', label='Agent 2', yerr=confs[:,2])
ax.bar(x + (3 * width), means[:,3], width, color='#4493F5', label='Agent 3', yerr=confs[:,3])
ax.bar(x + (4 * width), means[:,4], width, color='#2255F5', label='Agent 4', yerr=confs[:,4])



ax.set_ylabel('Average QE')
ax.set_ylim(0,0.200)
ax.set_xticks(x + width + width/2)
ax.set_xticklabels(x_labels)
ax.set_xlabel('Iteration Limits')
ax.set_title('Comparison of QE across iteration limits: Mesh Topology')
ax.legend()
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)


for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(16)
fig.tight_layout()
plt.show()










