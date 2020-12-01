import numpy as np
import matplotlib.pyplot as plt


def getInput(inpDic):
    # return self.data
    return np.asarray(list(inpDic.values()))  # A bit of a complex conversion for making sure what is returing is passable to train


def getInputIdx(inpDic):
    return list(inpDic.keys())

SAMPLES = 100
colors = np.random.rand(100,3)
idx1 = np.random.choice(colors.shape[0], 10, replace=False)
idx2 = np.random.choice(colors.shape[0], 10, replace=False)

idx1 = list(idx1)
idx2 = list(idx2)

col1 = colors[idx1,:]
col2 = colors[idx2,:]

inpDic1 = dict(zip(idx1, col1.tolist()))
inpDic2 = dict(zip(idx2, col2.tolist()))

keys = getInputIdx(inpDic1)
values = getInput(inpDic1)

inpList = [inpDic1, inpDic2]
both_keys = set(list(inpDic1.keys()) + list(inpDic2.keys()))


#
# all_gray = np.ones((100, 3))
# all_gray = all_gray*0.5
# all_gray[keys] = values
# mymap = all_gray.reshape(10, 10, 3)
#
#
# plt.imshow((mymap), interpolation='none')
# plt.title('Input')
# plt.show()