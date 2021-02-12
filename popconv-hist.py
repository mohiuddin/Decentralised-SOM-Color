import math
from minisom.minisom import MiniSom
from my_pkg.somagent import SomAgent
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.stats as st
import pandas as pd


def fixed_decay(learning_rate, t, max_iter):
    """This is a fixed decay custom fuction added by Rafi
    """
    return learning_rate

def main_loop(MEETING_LIMIT, certainlyMeet, agent, data, ITERATION_INPUT):
    N_AGENTS = len(agent)
    whCounter = 0
    while (whCounter < MEETING_LIMIT):
        print("** =========================================== **")
        # Communicate
        chancetoMeet = np.random.rand(4)
        if chancetoMeet[0] >= 0.5 or certainlyMeet:
            print("Agent 1 and 2 are communicating")
            agent[0].updateComm(agent[1], time.ctime())
        if chancetoMeet[1] >= 0.5 or certainlyMeet:
            print("Agent 2 and 3 are communicating")
            agent[1].updateComm(agent[2], time.ctime())
        if chancetoMeet[2] >= 0.5 or certainlyMeet:
            print("Agent 3 and 4 are communicating")
            agent[2].updateComm(agent[3], time.ctime())
        if chancetoMeet[3] >= 0.5 or certainlyMeet:
            print("Agent 4 and 1 are communicating")
            agent[3].updateComm(agent[0], time.ctime())

        # Getting more values from Environment
        for i in range(N_AGENTS):
            agent[i].updateEnv(data, ITERATION_INPUT)

        # Now training with the acquired inputs
        for i in range(N_AGENTS):
            agentInput = agent[i].getLastInputs()  # Getting unique values from all the recently acquired inputs
            # print(len(agentInput))
            agentInputToTrain = agent[i].samplesToTrain(
                agentInput)  # Getting the samples that have not been seen by the SOM
            # print(len(agentInputToTrain))
            l = len(agentInputToTrain)
            if (l > 0):
                print("Agent", i + 1, "Training SOM with", l, "samples")
                agent[i].som.train_batch(agentInputToTrain, len(agentInputToTrain), verbose=False)
            else:
                print("No new input for Agent", i)

        whCounter += 1

def varinterval(alpha, inputs , weights):
    n1 = len(inputs)
    n2 = len(weights)

    s1 = np.var(inputs, ddof=1)
    s2 = np.var(weights, ddof=1)

    F = st.f.ppf(1-alpha/2, n1-1, n2-1)
    lowerlim = (s1/s2)*(1/F)
    upperlim = (s1/s2)*F
    limits = [lowerlim, upperlim]

    return limits

def meaninterval(alpha, inputs, weights):
    n1 = len(inputs)
    n2 = len(weights)

    s1 = np.var(inputs, ddof=1)
    s2 = np.var(weights, ddof=1)
    x1 = np.mean(inputs)
    x2 = np.mean(weights)

    Z = st.norm.ppf(1-alpha/2)
    zFactor = Z * ((s1/n1+s2/n2)**0.5)

    lowerlim = (x1-x2)-zFactor
    upperlim = (x1-x2)+zFactor
    limits = [lowerlim, upperlim]

    return limits

def convergencescore(alpha, inputs, weights):

    #Getting the dimensions
    a = inputs.shape[1]
    b = weights.shape[1]
    if a != b:
        raise ValueError("Inputs and Weights dimesnions must be same!")
    conflags = np.zeros(a)
    #Putting each feature through mean and varaince convergence check
    for i in range(0,a):
        variance = varinterval(alpha, inputs[:,i], weights[:,i])
        mn = meaninterval(alpha, inputs[:,i], weights[:,i])
        if (variance[0] <= 1 <= variance[1]) and (mn[0] <= 0 <= mn[1]):
            conflags[i] = 1
        #     print("Feature",i,"has converged!")
        # if variance[0] <= 1 <= variance[1]:
        #     print("Variance has converged for feature", i)
        # if mn[0] <= 0 <= mn[1]:
        #     print("Mean has converged for feature", i)

    conscore = sum(conflags)/len(conflags)
    return conscore

def createAgents(mydata):
    soms = []
    agent = []
    for i in range(N_AGENTS):
        s = MiniSom(xdim, ydim, data_dim, sigma=sigma, learning_rate=lrate, decay_function=fixed_decay, neighborhood_function='bubble')
        #s = MiniSom(xdim, ydim, data_dim, sigma=1.0, learning_rate=1)
        #s.random_weights_init(mydata)
        #s.pca_weights_init(mydata)
        soms.append(s)
        a = SomAgent(i + 1, s)
        agent.append(a)
    return agent

def inputPrep():
    np.random.shuffle(mydata)
    for i in range(N_AGENTS):
        idx = np.random.choice(mydata.shape[0], INITIAL_INPUT, replace=False)  # Generating a certain no. of inputs for initial input
        colInput = mydata[idx, :]
        agent[i].inputInit(idx, colInput)

def trainCentral(mydata):
    csom = MiniSom(xdim, ydim, data_dim, sigma=sigma, learning_rate=lrate, decay_function=fixed_decay, neighborhood_function='bubble')
    #csom.random_weights_init(mydata)
    #csom = MiniSom(xdim, ydim, data_dim, sigma=1.0,learning_rate=1)
    #csom.pca_weights_init(mydata)
    csom.train_batch(mydata, len(mydata), verbose=True)
    return csom
'''
Data Preparation
'''
datapath = "C:\\Users\\Rafi\\Downloads\\data-05-00013-s001\\Hepta.csv"
df = pd.read_csv(datapath)
mydata = df.to_numpy()
mydata = mydata[:,1:4]
SAMPLES = len(mydata)
#mydata = (mydata - mydata.min(0)) / mydata.ptp(0)  #Normalise or not!
'''Colors Override'''
SAMPLES = 1600
mydata = np.random.rand(SAMPLES,3)*2-1
#mydata = mydata+1


''' Magic Numbers'''
INITIAL_INPUT = round(0.1*SAMPLES)   #10% of the total inputs
ITERATION_INPUT = round(0.1*SAMPLES)         #10% of the total inputs
N_AGENTS = 4
MEETING_LIMIT = 8
PLOTTING = False
certainlyMeet = False

''' More Magic Numbers'''
neurons = 5 * math.sqrt(SAMPLES)  # Using the heuristics: N = 5*sqrt(M)
xdim = round(math.sqrt(neurons))+1
ydim = round(neurons / xdim)+1
sigma = 1
lrate = 0.25
data_dim = 3
alpha = 0.05

''' Result Array'''
TRIALS = 1
totcolumns = 6
qresult = np.zeros((TRIALS,5))   #TRIALSx N_AGENTS+1 ARRAY TO HOLD THE RESULTS
cresult = np.zeros((TRIALS,5))

for k in range(TRIALS):
    agent = createAgents(mydata)
    inputPrep()
    main_loop(MEETING_LIMIT, certainlyMeet, agent, mydata, ITERATION_INPUT)
    csom = trainCentral(mydata)

    for i in range(N_AGENTS):
        inpDic = agent[i].inpDic
        input = list(inpDic.values())
        input = np.array(input)
        weights = agent[i].som.get_weights()

        sh = weights.shape
        elements = sh[0] * sh[1]
        eldim = sh[2]
        resshape_weights = np.reshape(weights, (elements, eldim))
        cscore = convergencescore(alpha, input, resshape_weights)
        cresult[k,i] = cscore
        qresult[k,i] = agent[i].som.quantization_error(input)

        i1 = list(input[:,0])
        i2 = list(input[:,1])
        i3 = list(input[:,2])

        w1 = list(resshape_weights[:,0])
        w2 = list(resshape_weights[:,1])
        w3 = list(resshape_weights[:,2])

        if i == 0:
            a = [i1, w1, i2, w2, i3, w3]
        elif i ==1:
            b = [i1, w1, i2, w2, i3, w3]
        elif i == 2:
            c = [i1, w1, i2, w2, i3, w3]
        elif i == 3:
            d = [i1, w1, i2, w2, i3, w3]

    plt.subplot(2,6,1)
    plt.hist(i1)
    plt.title("Input1")
    plt.subplot(2,6,2)
    plt.hist(w1)
    plt.title("Weight1")
    plt.subplot(2, 6, 3)
    plt.hist(i2)
    plt.title("Input2")
    plt.subplot(2, 6, 4)
    plt.hist(w2)
    plt.title("Weight2")
    plt.subplot(2, 6, 5)
    plt.hist(i3)
    plt.title("Input3")
    plt.subplot(2, 6, 6)
    plt.hist(w3)
    plt.title("Weight3")

    ''' For Central Now'''
    input = mydata
    weights = csom.get_weights()
    sh = weights.shape
    elements = sh[0]*sh[1]
    eldim = sh[2]
    resshape_weights = np.reshape(weights, (elements, eldim))
    cscore = convergencescore(alpha, input, resshape_weights)
    cresult[k,i+1] = cscore
    qresult[k,i+1] = csom.quantization_error(input)

    i1 = list(input[:, 0])
    i2 = list(input[:, 1])
    i3 = list(input[:, 2])
    w1 = list(resshape_weights[:, 0])
    w2 = list(resshape_weights[:, 1])
    w3 = list(resshape_weights[:, 2])
    e = [i1, w1, i2, w2, i3, w3]

print("==============** Results ** =========================\n")
print(np.mean(qresult, axis=0))
print(np.mean(cresult, axis=0))
#
''' Prepping data for boxplot'''

plt.subplot(2,6,7)
plt.hist(i1)
plt.title("Input1")
plt.subplot(2,6,8)
plt.hist(w1)
plt.title("Weight1")
plt.subplot(2, 6, 9)
plt.hist(i2)
plt.title("Input2")
plt.subplot(2, 6, 10)
plt.hist(w2)
plt.title("Weight2")
plt.subplot(2, 6, 11)
plt.hist(i3)
plt.title("Input3")
plt.subplot(2, 6, 12)
plt.hist(w3)
plt.title("Weight3")

plt.suptitle("Histogram: First row for one of the DSOMs, second row for Centralised")
plt.tight_layout()
plt.show()







