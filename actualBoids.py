import math
from minisom.minisom import MiniSom
from my_pkg.somagent import SomAgent
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.preprocessing import scale

def fixed_decay(learning_rate, t, max_iter):
    """This is a fixed decay custom fuction added by Rafi
    """
    return learning_rate

def ring_topology(agent):
    N = len(agent)
    chancetoMeet = np.random.rand(N)

    for i in range(N):
        if i == N-1:
            if chancetoMeet[i] >= 0.5:
                print("Agent %d and 1 are communicating" %(i+1))
                agent[i].updateComm(agent[0])
        else:
            if chancetoMeet[i] >= 0.5:
                print("Agent %d and %d are communicating" %(i+1,i+2))
                agent[i].updateComm(agent[i+1])
    return agent

def star_topology(agent):
    N =  len(agent)
    chancetoMeet = np.random.rand(N)

    for i in range(1,N):
        if chancetoMeet[i] >= 0.5:
            print("Agent 1 and %d are communicating" %(i+1))
            agent[0].updateComm(agent[i])

    return agent

def mesh_topology(agent):
    N = len(agent)
    meshlist = list(itertools.combinations(range(N),2))
    M = len(meshlist)
    chancetoMeet = np.random.rand(M)

    for i in range(M):
        if chancetoMeet[i] >= 0.5:
            a = meshlist[i][0]
            b = meshlist[i][1]
            print("Agent %d and %d are communicating" %(a+1,b+1))
            agent[a].updateComm(agent[b])

    return agent

def dsom_loop(meetlimit, agent, data, itrinput, topo='ring'):

    nagents = len(agent)
    whCounter = 0
    while (whCounter < meetlimit):
        print("** ============ Communicating ===================== **")

        if topo == 'ring':
            agent = ring_topology(agent)
        elif topo == 'mesh':
            agent = mesh_topology(agent)
        elif topo == 'star':
            agent = star_topology(agent)

        # Getting more values from Environment
        for i in range(nagents):
            agent[i].updateEnv(data, itrinput)

        # Now training with the acquired inputs
        for i in range(nagents):
            agentInput = agent[i].getLastInputs()  # Getting unique values from all the recently acquired inputs
            # print(len(agentInput))
            agentInputToTrain = agent[i].samplesToTrain(
                agentInput)  # Getting the samples that have not been seen by the SOM
            # print(len(agentInputToTrain))
            l = len(agentInputToTrain)
            if (l > 0):
                print("Agent", i + 1, "Training SOM with", l, "samples")
                agent[i].som.train_batch(agentInputToTrain, len(agentInputToTrain), verbose=True)
            else:
                print("No new myinput for Agent", i)

        whCounter += 1

    return agent

def kstest(myinput, weights):

    shw = weights.shape
    elements = shw[0] * shw[1]
    eldim = shw[2]
    resshape_weights = np.reshape(weights, (elements, eldim))

    shi = myinput.shape
    indim = shi[1]

    if eldim != indim:
        print("Erros in ks test- dimension must be same!")

    ksresults = np.zeros(eldim)
    for i in range(eldim):
        inp = list(myinput[:, i])
        wei = list(resshape_weights[:, i])
        temp  = st.ks_2samp(inp, wei)
        ksresults[i] = temp[1]

    return ksresults

def kstestweights(weight1, weight2):
    sh1 = weight1.shape
    elements1 = sh1[0] * sh1[1]
    eldim1 = sh1[2]
    resw1 = np.reshape(weight1, (elements1, eldim1))

    sh2 = weight2.shape
    elements2 = sh2[0] * sh2[1]
    eldim2 = sh2[2]
    resw2 = np.reshape(weight2, (elements2, eldim2))

    if eldim1 != eldim2:
        print("Erros in ks test- dimension must be same!")

    ksresults = np.zeros(eldim1)
    for i in range(eldim1):
        temp1 = list(resw1[:, i])
        temp2 = list(resw2[:, i])
        temp3 = st.ks_2samp(temp1, temp2)
        ksresults[i] = temp3[1]

    return ksresults


def inputPrep(agent, mydata, INITIAL_INPUT):
    l = len(agent)
    for i in range(l):
        idx = np.random.choice(mydata.shape[0], INITIAL_INPUT, replace=False)  # Generating a certain no. of inputs for initial myinput
        inputIdx = mydata[idx, :]
        agent[i].inputInit(idx, inputIdx)
    return agent

def createAgents(N_AGENTS, xdim,ydim, data_dim, sigma, lrate):
    soms = []
    agent = []
    for i in range(N_AGENTS):
        s = MiniSom(xdim, ydim, data_dim, sigma=sigma, learning_rate=lrate, decay_function=fixed_decay, neighborhood_function='bubble')
        soms.append(s)
        a = SomAgent(i + 1, s)
        agent.append(a)
    return agent

def initCentral(xdim, ydim, data_dim, sigma, lrate):
    csom = MiniSom(xdim, ydim, data_dim, sigma=sigma, learning_rate=lrate, decay_function=fixed_decay,
                   neighborhood_function='bubble')
    return csom

def trainCentral(csom, mydata):
    csom.train_batch(mydata, len(mydata), verbose=True)
    return csom

def makePlot():
    pass

def dataInit():
    pass

def main():
    '''
    Data Preparation
    '''
    datapath = "C:\\spike\\1\\initial.txt"
    df2 = pd.read_csv(datapath, header=None)
    print(df2.head(10))
    mydata = df2.to_numpy()
    T = 1 #Number of time steps sampled from the dataset, out of 1500
    Ts = T*200
    mydata = mydata[0:Ts,0:6]
    print(mydata)
    SAMPLES = len(mydata)

    ''' Magic Numbers :  Agent Related Settings'''
    toplgy = 'ring'
    INITIAL_INPUT = round(0.1*SAMPLES)           #10% of the total inputs
    ITERATION_INPUT = round(0.1*SAMPLES)         #5% of the total inputs
    N_AGENTS = 200
    MEETING_LIMIT = 4
    plotting = False
    ''' More Magic Numbers - SOM Parameters'''
    neurons = 5 * math.sqrt(SAMPLES)            # Using the heuristics: N = 5*sqrt(M)
    xdim = round(math.sqrt(neurons))+1
    ydim = round(neurons / xdim)+1
    sigma = 1
    lrate = 0.25
    data_dim = mydata.shape[1]
    TRIALS = 1

    ''' Result Array'''
    totcolumns = data_dim                                        #SOMS X no. of data dimensions
    dsomstats = np.zeros((TRIALS, totcolumns+1, N_AGENTS))       #One extra column for sample numbers
    cenvsdsom = np.zeros((TRIALS, totcolumns, N_AGENTS))         #This is for comparing dsom and central weights
    qEVals = np.zeros((TRIALS,N_AGENTS+1))        #For storing QE values

    #For trackign initial ks scores, before training
    csomstats = np.zeros((TRIALS, totcolumns))

    for k in range(TRIALS):

        ''' For Central'''
        ''' First getting the initial resutls - Turned off for now'''
        # weights = csom.get_weights()
        # res = kstest(mydata, weights)
        # initcsomstats[k, :] = res

        ''' Now training the centralised som and getting the results'''
        csom = initCentral(xdim, ydim, data_dim, sigma, lrate)
        csom = trainCentral(csom, mydata)
        cweights = csom.get_weights()
        res = kstest(mydata, cweights)
        csomstats[k, :] = res
        qEVals[k,0] = csom.quantization_error(mydata)

        '''For the DSOMs '''
        agent = createAgents(N_AGENTS, xdim, ydim, data_dim, sigma, lrate)
        agent = inputPrep(agent, mydata, INITIAL_INPUT)
        agent = dsom_loop(MEETING_LIMIT, agent, mydata, ITERATION_INPUT,toplgy)

        # Getting final ks values
        for i in range(N_AGENTS):
            inpDic = agent[i].inpDic
            dweights = agent[i].som.get_weights()
            res = kstest(mydata, dweights)
            dsomstats[k,range(data_dim),i] = res
            dsomstats[k,data_dim+1-1,i] = len(inpDic)
            qEVals[k,i+1] = agent[i].som.quantization_error(mydata)

            '''Now comapring denctralised vs centralised weights'''
            res = kstestweights(cweights, dweights)
            cenvsdsom[k, range(data_dim),i] = res

    print("==============** Results ** =========================\n")

    s1d = np.atleast_1d(SAMPLES)
    a = (np.mean(csomstats, axis=0))
    a = np.concatenate([a,s1d])



    ''' Doing the mean for the DSOMs'''
    dsomresults = np.zeros((N_AGENTS, totcolumns+1))
    cenvsdsomresults = np.zeros((N_AGENTS, totcolumns))
    for i in range(N_AGENTS):
        temp = np.mean(dsomstats[:,:,i], axis=0)
        dsomresults[i,:] = temp

        temp2 = np.mean(cenvsdsom[:,:,i], axis=0)
        cenvsdsomresults[i,:] = temp2

    dsomresults = np.row_stack([a, dsomresults])
    #print(dsomresults)
    print("==================")
    #print(cenvsdsomresults)

    df1 = pd.DataFrame(dsomresults)
    df2 = pd.DataFrame(cenvsdsomresults)

    df1 = df1.round(3)
    df2 = df2.round(3)

    with pd.ExcelWriter('test1.xlsx') as writer:
        df1.to_excel(writer, sheet_name='Sh1')
        df2.to_excel(writer, sheet_name='Sh2')

    if(plotting):
        makePlot()
if __name__ == "__main__":
    main()








