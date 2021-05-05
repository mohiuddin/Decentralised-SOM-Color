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
            print("Agent 1 is communicating with others")
            agent[0].updateComm(agent[1], time.ctime())
            agent[0].updateComm(agent[2], time.ctime())
            agent[0].updateComm(agent[3], time.ctime())

        if chancetoMeet[1] >= 0.5 or certainlyMeet:
            print("Agent 2 is communicating with others")
            agent[1].updateComm(agent[0], time.ctime())
            agent[1].updateComm(agent[2], time.ctime())
            agent[1].updateComm(agent[3], time.ctime())

        if chancetoMeet[2] >= 0.5 or certainlyMeet:
            print("Agent 3 is communicating with others")
            agent[2].updateComm(agent[0], time.ctime())
            agent[2].updateComm(agent[1], time.ctime())
            agent[2].updateComm(agent[3], time.ctime())

        if chancetoMeet[3] >= 0.5 or certainlyMeet:
            print("Agent 4 is communicating with others")
            agent[3].updateComm(agent[0], time.ctime())
            agent[3].updateComm(agent[1], time.ctime())
            agent[3].updateComm(agent[2], time.ctime())

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
                print("No new myinput for Agent", i)

        whCounter += 1

def kstest(input, weights):
    '''Takes as myinput two numpy arrays
    Right now works for only 3 dimensional data
    Returns an array with the three p-values for the three features'''
    sh = weights.shape
    elements = sh[0] * sh[1]
    eldim = sh[2]
    resshape_weights = np.reshape(weights, (elements, eldim))
    i1 = list(input[:, 0])
    i2 = list(input[:, 1])
    i3 = list(input[:, 2])
    w1 = list(resshape_weights[:, 0])
    w2 = list(resshape_weights[:, 1])
    w3 = list(resshape_weights[:, 2])
    ksresults1 = st.ks_2samp(i1, w1)
    ksresults2 = st.ks_2samp(i2, w2)
    ksresults3 = st.ks_2samp(i3, w3)
    res = [ksresults1[1], ksresults2[1], ksresults3[1]]
    return res

def createAgents(mydata):
    soms = []
    agent = []
    for i in range(N_AGENTS):
        s = MiniSom(xdim, ydim, data_dim, sigma=sigma, learning_rate=lrate, decay_function=fixed_decay, neighborhood_function='bubble')
        #s.random_weights_init(mydata)
        # s.pca_weights_init(mydata)
        soms.append(s)
        a = SomAgent(i + 1, s)
        agent.append(a)
    return agent

def inputPrep():
    for i in range(N_AGENTS):
        idx = np.random.choice(mydata.shape[0], INITIAL_INPUT, replace=False)  # Generating a certain no. of inputs for initial myinput
        inputIdx = mydata[idx, :]
        agent[i].inputInit(idx, inputIdx)


def initCentral():
    csom = MiniSom(xdim, ydim, data_dim, sigma=sigma, learning_rate=lrate, decay_function=fixed_decay,
                   neighborhood_function='bubble')
    #csom.random_weights_init(mydata)
    # csom.pca_weights_init(mydata)
    return csom

def trainCentral(csom, mydata):
    csom.train_batch(mydata, len(mydata), verbose=True)
    return csom
'''
Data Preparation
'''
datapath = "C:\\Users\\Rafi\\Downloads\\data-05-00013-s001\\Tetra.csv"
df = pd.read_csv(datapath)
mydata = df.to_numpy()
mydata = mydata[:,1:4]
SAMPLES = len(mydata)

'''Colors Override'''
SAMPLES = 10000
mydata = np.random.rand(SAMPLES,3)*2-1


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

''' Result Array'''
TRIALS = 10
totcolumns = 3 #SOMS X 3 columns each
dsom1 = np.zeros((TRIALS, totcolumns))
dsom2 = np.zeros((TRIALS, totcolumns))
dsom3 = np.zeros((TRIALS, totcolumns))
dsom4 = np.zeros((TRIALS, totcolumns))
csomres = np.zeros((TRIALS, totcolumns))

''' For tracking the number of samples seen by dsoms'''
dsomsamples = np.zeros((TRIALS,4))  #Hard coded for four agents now

#For trackign initial ks scores, before training
initdsom1 = np.zeros((TRIALS, totcolumns))
initdsom2 = np.zeros((TRIALS, totcolumns))
initdsom3 = np.zeros((TRIALS, totcolumns))
initdsom4 = np.zeros((TRIALS, totcolumns))
initcsomres = np.zeros((TRIALS, totcolumns))

for k in range(TRIALS):
    agent = createAgents(mydata)
    csom = initCentral()
    inputPrep()
    '''This part is to get an idea whether ks test values are changing before
        and after training'''
    for i in range(N_AGENTS):
        inpDic = agent[i].inpDic
        input = list(inpDic.values())
        input = np.array(input)
        weights = agent[i].som.get_weights()

        res = kstest(mydata, weights)

        if i == 0:
            initdsom1[k,:] = res
        elif i ==1:
            initdsom2[k,:] = res
        elif i == 2:
            initdsom3[k, :] = res
        elif i == 3:
            initdsom4[k, :] = res
    main_loop(MEETING_LIMIT, certainlyMeet, agent, mydata, ITERATION_INPUT)

    for i in range(N_AGENTS):
        inpDic = agent[i].inpDic
        input = list(inpDic.values())
        input = np.array(input)
        weights = agent[i].som.get_weights()

        res = kstest(mydata, weights)

        if i == 0:
            dsom1[k,:] = res
            dsomsamples[k,0] = len(inpDic)
        elif i ==1:
            dsom2[k,:] = res
            dsomsamples[k, 1] = len(inpDic)
        elif i == 2:
            dsom3[k, :] = res
            dsomsamples[k, 2] = len(inpDic)
        elif i == 3:
            dsom4[k, :] = res
            dsomsamples[k, 3] = len(inpDic)

    ''' For Central Now'''
    ''' First getting the initial resutls'''
    weights = csom.get_weights()
    res = kstest(mydata, weights)
    initcsomres[k,:] = res

    ''' Now training the centralised som and getting the results'''
    csom = trainCentral(csom, mydata)
    weights = csom.get_weights()
    res = kstest(mydata, weights)
    csomres[k,:] = res

print("==============** Results ** =========================\n")

x = (np.mean(initcsomres, axis=0))
y = (np.mean(initdsom1, axis=0))
z = (np.mean(initdsom2, axis=0))
p = (np.mean(initdsom3, axis=0))
q = (np.mean(initdsom4, axis=0))


X = (np.concatenate([x,y,z,p,q]))
#np.savetxt('test.csv', z)
#print(X)


a = (np.mean(csomres, axis=0))
b = (np.mean(dsom1, axis=0))
c = (np.mean(dsom2, axis=0))
d = (np.mean(dsom3, axis=0))
e = (np.mean(dsom4, axis=0))


A = (np.concatenate([a,b,c,d,e]))
C = np.row_stack((X, A))
print(C)

snumbers = pd.DataFrame((np.mean(dsomsamples, axis = 0)))
dftowrite = pd.DataFrame(C)

with pd.ExcelWriter('test1.xlsx') as writer:
    dftowrite.to_excel(writer, sheet_name='Sh1')
    snumbers.to_excel(writer, sheet_name='Sh2')









