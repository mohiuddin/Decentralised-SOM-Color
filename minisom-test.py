import math
from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt

# Generating Random Color Data
samples = 1600
colors_2D = np.random.rand(40,40,3)
colors = colors_2D.ravel()
colors = colors.reshape(1600,3)

# SOM parameters
neurons = 5 * math.sqrt(samples)  # Using the heuristics: N = 5*sqrt(M)
xdim = math.ceil(math.sqrt(neurons))
ydim = math.ceil(neurons / xdim)
sigma = 1
learning_rate = 0.25

# Draw Input

plt.subplot(131)
plt.imshow(abs(colors_2D),interpolation='none')
plt.title('Input')


# Initialize SOM
som1 = MiniSom(xdim, ydim, 3)
# som1.pca_weights_init(colors)
som1init = som1
plt.subplot(132)
plt.imshow(abs(som1init.get_weights()), interpolation='none')
plt.title("Initial")

# Train SOM
som1.train(colors, 1000, verbose=True)

# Draw Final SOM
plt.subplot(133)
plt.imshow(abs(som1.get_weights()), interpolation='none')
qE1 = som1.quantization_error(colors)
plt.title("Final," + " QE = " + str(round(qE1, 2)))
plt.show()
