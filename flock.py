import math
from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
import time

class SomAgent:
    def __init__(self, agent_id, som):
        self.ID = agent_id
        self.som = som
        self.commHistory = []
        print("Agent", self.ID, "created")

    def updateComm(self, fromAgent , newComm):
        updated = [fromAgent, newComm]
        self.commHistory.append(updated)

    def printComm(self):
        print("Communication history for Agent ID:", self.ID)
        print(self.commHistory)
    def inputTracker(self,input,idx):
        self.initInput = input
        self.initID = idx
        self.input = input

    def getInput(self):
        return self.input

#####################################################################################################

def convto2d (colors):
    size1d = colors.shape[0]
    size2d = int(math.sqrt(size1d))
    cols = colors.reshape(size2d, size2d, 3)
    return cols

def convto1d (colors):
    sizex = colors.shape[0]
    sizey = colors.shape[1]
    cols = colors.reshape(sizex*sizey,3)
    return cols


# Generating Random Color Data
samples = 1600
colors_2D = np.random.rand(40,40,3)
colors = convto1d(colors_2D)

# Agent numbers
n_agents = 4
soms = []
agent = []

# SOM parameters
neurons = 5 * math.sqrt(samples)  # Using the heuristics: N = 5*sqrt(M)
xdim = round(math.sqrt(neurons))
ydim = round(neurons / xdim)
sigma = 1
lrate = 0.25
def fixed_decay(learning_rate, t, max_iter):
    """This is a fixed decay custom fuction
    added by Rafi
    """
    return learning_rate

# Creating agents in a loop
for i in range(n_agents):
    s = MiniSom(xdim, ydim, 3, sigma = sigma, learning_rate=lrate, decay_function = fixed_decay)
    soms.append(s)
    a = SomAgent(i+1,s)
    agent.append(a)

# Creating Randomized Input Subsets
colInputs = np.zeros((400, 3))
for i in range(n_agents):
    idx = np.random.choice(colors.shape[0], 400, replace=True)   #Generating 400 random indexe with replacement
    colInputs = colors[idx,:]
    agent[i].inputTracker(colInputs, idx)

# The Main Loop

meetFlag = [0, 0, 0, 0]
while (True):
    #Draw Plots
    plotCounter = 1
    for i in range(n_agents):

        plt.subplot(4, 3, plotCounter)
        agentInput = agent[i].getInput();
        agentInput2d = convto2d(agentInput)
        plt.imshow(abs(agentInput2d), interpolation='none')
        plt.title('Input')
        plotCounter = plotCounter+1

        plt.subplot(4, 3, plotCounter)
        plt.imshow(abs(agent[i].som.get_weights()), interpolation='none')
        plt.title("Initial MAP")
        plotCounter = plotCounter + 1

        # Training the SOM
        agent[i].som.train_batch(agentInput, samples, verbose=True)

        # After Training
        plt.subplot(4, 3, plotCounter)
        plt.imshow(abs(agent[i].som.get_weights()), interpolation='none')
        qE1 = agent[i].som.quantization_error(colors)
        plt.title("Final," + " QE = " + str(round(qE1, 4)))
        plotCounter = plotCounter + 1

    plt.subplots_adjust(hspace=0.3)
    plt.show()

    #Communicate, which should include update

    chancetoMeet = np.random.rand(4)
    if chancetoMeet[0] > 0.5:
        print("Agent 1 and 2 are communicating")
        agent[0].updateComm(2,time.ctime())
        break
    if chancetoMeet[1] > 0.5:
        print("Agent 2 and 3 are communicating")
    if chancetoMeet[2] > 0.5:
        print("Agent 3 and 4 are communicating")
    if chancetoMeet[3] > 0.5:
        print("Agent 4 and 1 are communicating")




agent[0].printComm()







