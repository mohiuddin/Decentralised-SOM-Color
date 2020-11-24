import numpy as np


SAMPLES = 1600
colors = np.random.rand(10,3)
idx1 = np.random.choice(colors.shape[0], 3, replace=False)
idx2 = np.random.choice(colors.shape[0], 3, replace=False)

idx1 = list(idx1)
idx2 = list(idx2)

col1 = colors[idx1,:]
col2 = colors[idx2,:]


inpDic1 = dict(zip(idx1, col1.tolist()))
inpDic2 = dict(zip(idx2, col2.tolist()))


