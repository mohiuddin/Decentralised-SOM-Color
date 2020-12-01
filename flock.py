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
        self.envRatio = 0.1   #Percentage of the total sample size (Environment) it will traverse at each go
        self.inpList = []
        self.inputCount = 0
        self.currentCount = 0
        self.initFlag = True
        self.repository = set()
        print("Agent", self.ID, "created")

    def inputInit(self,idx, data):
        self.idx = idx
        self.data = data
        inpDic = dict(zip(idx, data.tolist() ))
        self.inpDic = inpDic
        self.inpList.append(inpDic)
        self.inputCount += 1


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
        self.inpList.append(tosend_o)
        self.inputCount += 1

        # Update other's dictionary
        other.inpDic.update(tosend_s)
        other.inpList.append(tosend_s)
        other.inputCount += 1

        # Updating the textual history of the communication
        updated = "Received " + str(items_o) + " items from Agent " + str(other.ID)
        self.commHistory.append(updated)
        updated = "Received " + str(items_s) + " items from Agent " + str(self.ID)
        other.commHistory.append(updated)

    def updateEnv (self, colors, ITERATION_INPUT):
        items = ITERATION_INPUT
        idx = np.random.choice(colors.shape[0], ITERATION_INPUT, replace=False)
        data = colors[idx, :]
        inpDicEnv = dict(zip(idx, data.tolist() ))

        self.inpList.append(inpDicEnv)   # Appending to the list of dictionary
        self.inpDic.update(inpDicEnv)    # Updating the whole dictionary, may not be useful
        self.inputCount += 1


    def printComm(self):
        print("Communication history for Agent:", self.ID)
        for i in range(0, len(self.commHistory)):
            print(self.commHistory[i])

    def getAllInput(self):
        #return self.data for the whole current state
        return np.asarray(list(self.inpDic.values()))   #A bit of a complex conversion for making sure what is returing is passable to train
    def getAllInputIdx(self):
        #Returns all the ids
        return list(self.inpDic.keys())

    def getLastInputs(self):
        current = self.currentCount
        last = self.inputCount
        if current == 0:
            lastGen = self.inpList
            increaseCounter = 1
        else:
            lastGen = self.inpList[current:last+1]
            increaseCounter = last-current

        self.currentCount += increaseCounter

        toSend = {}   #Creating empty dictionary
        #Updating with each of the dictionaries in the current generation
        #So it will have unique keys, with later keys replacign earlier ones
        if increaseCounter == 1:
            #print(type(lastGen[0]))
            toSend.update(lastGen[0])
        else:
            for i in range(0, len(lastGen)):
                toSend.update(lastGen[i])
        return toSend

    def samplesToTrain(self, currInput):
        currInputIdx = set(list(currInput.keys()))   #Get the IDs of the current input
        print("Length of passed input for Agent", self.ID+1, "is", len(currInputIdx))

        #This is the case when agent is training for the fist time
        if self.initFlag == True:
            toTrainIdx = currInputIdx
            self.repository = self.repository.union(currInputIdx)
            self.initFlag = False
            #print("When Init Flag is True")
            #print("Initial Repository length for Agent", self.ID+1,len(self.repository))

        else:
            toTrainIdx = currInputIdx - self.repository  # Get only the IDs that are not in repository already
            self.repository = self.repository.union(toTrainIdx)  # Update repository by adding the current IDs

        print("Final Repository length for Agent", self.ID+1,"is",len(self.repository))
        #Now building a dictionary with only the new values and returning the values
        toTrainIdx = list(toTrainIdx)
        print("Length of returned input", len(toTrainIdx))
        toTrainDict = {k: currInput[k] for k in toTrainIdx} #Taking only the selected entries fo the input and builda new dicitonary

        return np.asarray(list(toTrainDict.values()))

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
    keys  = a.getAllInputIdx()
    values = a.getAllInput()
    all_gray[keys] =  values
    mymap = all_gray.reshape(40, 40, 3)
    return mymap
###############################MAGIC Nummbers     ######################################
# Number of samples and agetns etc.
SAMPLES = 10000
colors_2D = np.random.rand(40,40,3)
colors = convto1d(colors_2D)
INITIAL_INPUT = 100
ITERATION_INPUT = 100

meetCounter = [0, 0, 0, 0]             # Counting the Meetings  (1,2) (2,3) (3,4) (4,1)
certainlyMeet = True                  # True if agents are guaranteed to meet at each iteration
inputPortion = 0.1                     # The percentage of existing input that is shared between agents when in contact

N_AGENTS = 4
MEETING_LIMIT = 28                  #This is the total meet count limit. Invividual counts are random
PLOTTING = False

# SOM parameters
neurons = 5 * math.sqrt(SAMPLES)       # Using the heuristics: N = 5*sqrt(M)
xdim = round(math.sqrt(neurons))
ydim = round(neurons / xdim)
sigma = 1
lrate = 0.25
data_dim = 3                            #Data dimension. RGB is 3-dimensional
###########################   End Magic Numbers    ##########################################################
def fixed_decay(learning_rate, t, max_iter):
    """This is a fixed decay custom fuction
    added by Rafi
    """
    return learning_rate

# Creating agents in a loop, each with a SOM object of its own
soms = []
agent = []
for i in range(N_AGENTS):
    s = MiniSom(xdim, ydim, data_dim, sigma = sigma, learning_rate=lrate, decay_function = fixed_decay)
    soms.append(s)
    a = SomAgent(i+1,s)
    agent.append(a)

# Creating Randomized Input Subsets for Initialisation of the scheme
colInput = np.zeros((INITIAL_INPUT, data_dim))
for i in range(N_AGENTS):
    idx = np.random.choice(colors.shape[0], INITIAL_INPUT, replace=False)   #Generating a certain no. of inputs for initial input
    colInput = colors[idx,:]
    agent[i].inputInit(idx, colInput)   # This is the initial input

# The Main Loop
             # This is the portion of input that is getting communicated at each chance
while (sum(meetCounter) < MEETING_LIMIT):
    #Draw Plots
    plotCounter = 1
    ax = plt.subplot()
    for i in range(N_AGENTS):
        # plt.subplot(4, 3, plotCounter)
        # agentInput = agent[i].getAllInput()
        # agentInputPlot = convtoPlot(agent[i])
        #
        # plt.imshow((agentInputPlot), interpolation='none')
        # plt.title('Input')
        # plotCounter = plotCounter+1
        #
        # plt.subplot(4, 3, plotCounter)
        # plt.imshow(abs(agent[i].som.get_weights()), interpolation='none')
        # plt.title("Starting MAP")
        # plotCounter = plotCounter + 1

        # Training the SOM


        agentInput = agent[i].getLastInputs()                      #Getting unique values from all the recently acquired inputs
        #print(len(agentInput))
        agentInputToTrain = agent[i].samplesToTrain(agentInput)    #Getting the samples that have not been seen by the SOM
        #print(len(agentInputToTrain))
        l = len(agentInputToTrain)
        if (l > 0):
            print("Agent",i+1,"Training SOM with", l, "samples")
            agent[i].som.train_batch(agentInputToTrain, len(agentInputToTrain), verbose=False)
        else:
            print("No new input for Agent", i)


        # After Training
    #     plt.subplot(4, 3, plotCounter)
    #     plt.imshow(abs(agent[i].som.get_weights()), interpolation='none')
    #     qE1 = agent[i].som.quantization_error(colors)
    #     plt.title("Final, " + " QE = " + str(round(qE1, 4)))
    #     plotCounter = plotCounter + 1
    # plt.subplots_adjust(hspace=0.3)
    # # Show Plot if Flag is Turned ON
    # if (PLOTTING):
    #     plt.show()

    #Communicate, which should include update
    chancetoMeet = np.random.rand(4)
    if chancetoMeet[0] >= 0.5 or certainlyMeet:
        print("Agent 1 and 2 are communicating")
        agent[0].updateComm(agent[1],time.ctime())
        meetCounter[0] += 1

    if chancetoMeet[1] >= 0.5 or certainlyMeet:
        print("Agent 2 and 3 are communicating")
        agent[1].updateComm(agent[2], time.ctime())
        meetCounter[1] += 1

    if chancetoMeet[2] >= 0.5 or certainlyMeet:
        print("Agent 3 and 4 are communicating")
        agent[2].updateComm(agent[3], time.ctime())
        meetCounter[2] += 1

    if chancetoMeet[3] >= 0.5 or certainlyMeet:
        print("Agent 4 and 1 are communicating")
        agent[3].updateComm(agent[0], time.ctime())
        meetCounter[3] += 1

    # Getting more values from Environment
    for i in range(N_AGENTS):
        agent[i].updateEnv(colors, ITERATION_INPUT)



#Centralised SOM
csom = MiniSom(xdim, ydim, data_dim, sigma=sigma, learning_rate=lrate, decay_function=fixed_decay)
csom.train_batch(colors, len(colors), verbose=True)

#Generate Test Set

testLength = 10
qval = np.zeros((10, 5))
for i in range(0, testLength):

    testSet = np.random.rand(5000, 3)
    qec = csom.quantization_error(testSet)
    qe1 = agent[0].som.quantization_error(testSet)
    qe2 = agent[1].som.quantization_error(testSet)
    qe3 = agent[2].som.quantization_error(testSet)
    qe4 = agent[3].som.quantization_error(testSet)

    qval[i,0] = qec
    qval[i,1] = qe1
    qval[i,2] = qe2
    qval[i,3] = qe3
    qval[i,4] = qe4

print(qval)






