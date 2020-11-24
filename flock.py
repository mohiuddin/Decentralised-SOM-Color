import math
from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
import time
import random


################# Class Def ###########################################################################
class SomAgent:
    def __init__(self, agent_id, som):
        self.ID = agent_id
        self.som = som
        self.commHistory = []
        self.commRatio = 0.1   #Now fixed, can be initialised to a random value if decided
        print("Agent", self.ID, "created")

    def inputInit(self,idx, data):
        self.idx = idx
        self.data = data
        inpDic = dict(zip(idx, data.tolist() ))
        self.inpDic = inpDic

    def updateComm(self, other , tstamp):
        """They will pass with another only a random percentage of the data they have
        The textual history will also be updated"""

        #Check length
        inplength_s = len(self.inpDic)
        items_s  = int(inplength_s*self.commRatio)
        tosend_s = dict(random.sample(self.inpDic.items(), items_s))

        inplength_o = len(other.inpDic)
        items_o = int(inplength_o * other.commRatio)
        tosend_o = dict(random.sample(other.inpDic.items(), items_o))

        #Update self's dictionary
        self.inpDic.update(tosend_o)
        other.inpDic.update(tosend_s)
        #Update other's dictionary


        # Updating the textual history of the communication
        updated = "Received " + str(items_o) + " items from Agent" + str(other.ID) + " at " + tstamp
        self.commHistory.append(updated)
        updated = "Received " + str(items_s) + " items from Agent " + str(self.ID) + " at " + tstamp
        other.commHistory.append(updated)


    def printComm(self):
        print("Communication history for Agent:", self.ID)
        for i in range(0, len(self.commHistory)):
            print(self.commHistory[i])

    def getInput(self):
        #return self.data
        return np.asarray(list(self.inpDic.values()))   #A bit of a complex conversion for making sure what is returing is passable to train
    def getInputIdx(self):
        return list(self.inpDic.keys())


########## Helper Functions    ######################################################

def convto2d (colors):
    size1d = colors.shape[0]
    size2dx = int(math.sqrt(size1d))
    cols = colors.reshape(size2dx, -1, 3)
    return cols

def convto1d (colors):
    sizex = colors.shape[0]
    sizey = colors.shape[1]
    cols = colors.reshape(sizex*sizey,3)
    return cols

def convtoPlot(a):
    """Draw SAMPLES size grid with non-existing IDs as gray"""
    all_gray = np.ones((1600, 3))
    all_gray = all_gray*0.5
    keys  = a.getInputIdx()
    values = a.getInput()
    all_gray[keys] =  values
    mymap = all_gray.reshape(40, 40, 3)

    return mymap


#########################################################################################################
# Generating Random Color Data
SAMPLES = 1600
colors_2D = np.random.rand(40,40,3)
colors = convto1d(colors_2D)

# Agent numbers
N_AGENTS = 4
soms = []
agent = []

# SOM parameters
neurons = 5 * math.sqrt(SAMPLES)  # Using the heuristics: N = 5*sqrt(M)
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
for i in range(N_AGENTS):
    s = MiniSom(xdim, ydim, 3, sigma = sigma, learning_rate=lrate, decay_function = fixed_decay)
    soms.append(s)
    a = SomAgent(i+1,s)
    agent.append(a)

# Creating Randomized Input Subsets
colInput = np.zeros((400, 3))
for i in range(N_AGENTS):
    idx = np.random.choice(colors.shape[0], 400, replace=False)   #Generating 400 random index without replacement
    colInput = colors[idx,:]
    agent[i].inputInit(idx, colInput)   # This is the initial input

# The Main Loop

meetCounter = [0, 0, 0, 0]           # Counting the Meetings  (1,2) (2,3) (3,4) (4,1)
certainlyMeet = False                 # True if agents are guaranteed to meet at each iteration
inputPortion = 0.1                   # This is the portion of input that is getting communicated at each chance
while (sum(meetCounter) < 8):
    #Draw Plots
    plotCounter = 1
    for i in range(N_AGENTS):

        plt.subplot(4, 3, plotCounter)
        agentInput = agent[i].getInput()
        agentInputPlot = convtoPlot(agent[i])

        plt.imshow((agentInputPlot), interpolation='none')
        plt.title('Input')
        plotCounter = plotCounter+1

        plt.subplot(4, 3, plotCounter)
        plt.imshow(abs(agent[i].som.get_weights()), interpolation='none')
        plt.title("Starting MAP")
        plotCounter = plotCounter + 1

        # Training the SOM
        agent[i].som.train_batch(agentInput, len(agentInput), verbose=True)

        # After Training
        plt.subplot(4, 3, plotCounter)
        plt.imshow(abs(agent[i].som.get_weights()), interpolation='none')
        qE1 = agent[i].som.quantization_error(colors)
        plt.title("Final, " + " QE = " + str(round(qE1, 4)))
        plotCounter = plotCounter + 1

    plt.subplots_adjust(hspace=0.3)
    plt.show()

    #Communicate, which should include update

    chancetoMeet = np.random.rand(4)
    if chancetoMeet[0] > 0.5 or certainlyMeet:
        print("Agent 1 and 2 are communicating")
        agent[0].updateComm(agent[1],time.ctime())
        meetCounter[0] += 1

    if chancetoMeet[1] > 0.5 or certainlyMeet:
        print("Agent 2 and 3 are communicating")
        agent[1].updateComm(agent[2], time.ctime())
        meetCounter[1] += 1

    if chancetoMeet[2] > 0.5 or certainlyMeet:
        print("Agent 3 and 4 are communicating")
        agent[2].updateComm(agent[3], time.ctime())
        meetCounter[2] += 1

    if chancetoMeet[3] > 0.5 or certainlyMeet:
        print("Agent 4 and 1 are communicating")
        agent[3].updateComm(agent[0], time.ctime())
        meetCounter[3] += 1




agent[0].printComm()
agent[1].printComm()
agent[2].printComm()
agent[3].printComm()








