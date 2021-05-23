import math
from minisom.minisom import MiniSom
from my_pkg.somagent import SomAgent
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
import itertools
import networkx as nx
import pickle
from datetime import datetime
import collections
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
            agentInputToTrain = agent[i].samplesToTrain(agentInput)  # Getting the samples that have not been seen by the SOM
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

def boidInputPrep(agent, mydata):
    l = len(agent)
    for i in range(l):
        j = np.array([i])
        agent[i].boidInputInit(j,mydata[j,1:])
    return agent


def createAgents(N_AGENTS, xdim,ydim, data_dim, sigma, lrate):
    soms = []
    agent = []
    for i in range(N_AGENTS):
        s = MiniSom(xdim, ydim, data_dim, sigma=sigma, learning_rate=lrate, decay_function=fixed_decay, neighborhood_function='bubble')
        soms.append(s)
        a = SomAgent(i, s)
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


def toGraph(posdata, RADIUS):
    graphDict = {}
    chunks = np.array_split(posdata,1500)
    print(len(chunks))

    for i in range(0, len(chunks)):
        #print(i)
        thischunk = chunks[i]
        dist = np.abs(thischunk[np.newaxis, :, :] - thischunk[:, np.newaxis, :]).min(axis=2)
        adjMat = dist < RADIUS
        np.fill_diagonal(adjMat, 0)
        adjMat = adjMat.astype(int)
        #print(adjMat)
        G = nx.from_numpy_matrix(adjMat)
        graphDict[i] = nx.to_dict_of_lists(G)

    return graphDict


def boid_dsom_loop(agent, mydata, gDict):

    nagents = len(agent)
    start = 1
    end = 1500
    chunks = np.array_split(mydata, 1500)

    for t in range(start,end):

        #if agent is in neighbourhood, communicate
        print("Now starting timestep: ", t)

        adj = gDict[t]
        for i in range(nagents):
            neighlist = adj[i]                  #This is now a list
            for j in neighlist:
                #if j >= nagents:                 #Temporary, for debugging
                #    continue
                #else:
                    agent[i].boidUpdateComm(agent[j])

        for i in range(nagents):
            agent[i].boidUpdateEnv(chunks[t])

        for i in range(nagents):
            agentInput = agent[i].getLastInputs()  # Getting unique values from all the recently acquired inputs
            # print(len(agentInput))
            agentInputToTrain = agent[i].samplesToTrain(agentInput)  # Getting the samples that have not been seen by the SOM
            #agentInputToTrain = agentInputToTrain[:, 1:]             # Extra step to get rid of the first column
            #print(agentInputToTrain)
            l = len(agentInputToTrain)
            if (l > 0):
                print("Agent", i, "Training SOM with", l, "samples")
                agent[i].som.train_batch(agentInputToTrain, len(agentInputToTrain), verbose=True)
            else:
                print("No new myinput for Agent", i)


    return agent

def main():
    '''
    Data Preparation
    '''
    datapath = "C:\\spike\\1\\initial.txt"
    RADIUS = 100
    df = pd.read_csv(datapath, header=None)
    df.drop(df.columns[[0, 1, 6, 7, 8, 9, 10, 11, 12, 13]], axis=1, inplace=True)
    df2 = df.iloc[0:300000, :]
    newcol = np.arange(300000).transpose()
    df2.insert(0,0,newcol)


    mydatafile = df2.to_numpy()
    del df, df2

    posdata = mydatafile[:,1:3]         #Taking only position values
    #pri1nt(posdata[0:10])
    mydata = mydatafile[:,[0,3,4]]      #Taking the index and velocity columns
    #print(mydata[0:10])
    SAMPLES = len(mydata)


    print("Generating Adjacency list") #Takes about 30 seconds to do it
    graphDic = toGraph(posdata,RADIUS)
    print("Adjacency list generation complete!")


    # degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    # degreeCount = collections.Counter(degree_sequence)
    # deg, cnt = zip(*degreeCount.items())
    # fig, ax = plt.subplots()
    # plt.bar(deg, cnt, color="b")
    # plt.title("Degree Histogram")
    # plt.ylabel("Count")
    # plt.xlabel("Degree")
    # ax.set_xticks([d for d in deg])
    # ax.set_xticklabels(deg)
    # #ax.set_xticks(ax.get_xticks()[::3])
    # #plt.tick_params(axis='x', which='major', labelsize=6)
    # plt.show()


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
    boid_data_dim = data_dim-1                  #Because boid data has a unique ID as part of data
    TRIALS = 1

    ''' Result Array'''
    totcolumns = boid_data_dim                                        #SOMS X no. of data dimensions
    dsomstats = np.zeros((TRIALS, totcolumns+1, N_AGENTS))       #One extra column for sample numbers
    cenvsdsom = np.zeros((TRIALS, totcolumns, N_AGENTS))         #This is for comparing dsom and central weights
    qEVals = np.zeros((TRIALS,N_AGENTS+1))                       #For storing QE values
    # For trackign initial ks scores, before training
    csomstats = np.zeros((TRIALS, totcolumns))


    #agent = createAgents(N_AGENTS, xdim, ydim, boid_data_dim, sigma, lrate)
    #agent = boidInputPrep(agent, mydata)
    #agent = boid_dsom_loop(agent,mydata,graphDic)



    #
    #     agent = dsom_loop(MEETING_LIMIT, agent, mydata, ITERATION_INPUT, toplgy)

    for k in range(TRIALS):

        ''' For Central'''
        ''' First getting the initial resutls - Turned off for now'''
        # weights = csom.get_weights()
        # res = kstest(mydata, weights)
        # initcsomstats[k, :] = res

        ''' Now training the centralised som and getting the results'''
        csom = initCentral(xdim, ydim, boid_data_dim, sigma, lrate)
        csom = trainCentral(csom, mydata[:,1:])
        cweights = csom.get_weights()
        #res = kstest(mydata[:,1:], cweights)
        #csomstats[k, :] = res
        qEVals[k, 0] = csom.quantization_error(mydata[:,1:])

        '''For the DSOMs '''
        agent = createAgents(N_AGENTS, xdim, ydim, boid_data_dim, sigma, lrate)
        agent = boidInputPrep(agent, mydata)
        agent = boid_dsom_loop(agent, mydata, graphDic)
        # Getting final ks values
        for i in range(N_AGENTS):
            inpDic = agent[i].inpDic
            #dweights = agent[i].som.get_weights()
            #res = kstest(mydata[:,1:], dweights)
            #dsomstats[k, range(boid_data_dim), i] = res
            dsomstats[k, boid_data_dim + 1 - 1, i] = len(inpDic)
            qEVals[k, i + 1] = agent[i].som.quantization_error(mydata[:,1:])

            '''Now comapring denctralised vs centralised weights'''
            #res = kstestweights(cweights, dweights)
            #cenvsdsom[k, range(boid_data_dim), i] = res

        myfile = open('myagents.obj','wb')
        pickle.dump(agent,myfile)
        myfile2 = open('central.obj', 'wb')
        pickle.dump(csom,myfile2)


    print("==============** Results ** =========================\n")
    qEres = np.mean(qEVals, axis=0)
    print(qEres)

    df1 = pd.DataFrame(qEVals)
    df2 = pd.DataFrame(dsomstats)
    df1 = df1.round(3)
    df2 = df2.round(3)


    with pd.ExcelWriter('test1.xlsx') as writer:
        df1.to_excel(writer, sheet_name='Sh1')
        df2.to_excel(writer, sheet_name='Sh2')


if __name__ == "__main__":
    main()








