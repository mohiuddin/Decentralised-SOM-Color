import numpy as np
import random



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

    def boidInputInit(self, idx, data):
        #idx = []
        #idxtemp = int(dataset[0])
        #idx.append(idxtemp)

        #data = [[]]
        #datatemp = dataset[1:3].tolist()
        #data = data.append(datatemp)
        #print(data)
        #data2 = [x for x in data if x!= []]


        self.idx = idx
        self.data = data
        inpDic = dict(zip(idx, data.tolist()))
        #print(idx)
        #print(data)
        print(inpDic)
        #print(type(data))
        self.inpDic = inpDic
        self.inpList.append(inpDic)
        self.inputCount += 1


    def updateComm(self, other, verbose=False):
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
        updated = "\tAgent "+ str(self.ID) + " received " + str(items_o) + " items from Agent " + str(other.ID)
        self.commHistory.append(updated)
        if(verbose):
            print(updated)
        updated = "\tAgent " + str(other.ID)+ " received " + str(items_s) + " items from Agent " + str(self.ID)
        other.commHistory.append(updated)
        if(verbose):
            print(updated)

    def boidUpdateComm(self, other, verbose=True):
        """They will pass with another only a random percentage of the data they have
        The textual history will also be updated"""
        # Check length
        inplength_s = len(self.inpDic)
        tosend_s = dict(random.sample(self.inpDic.items(), inplength_s))
        inplength_o = len(other.inpDic)
        tosend_o = dict(random.sample(other.inpDic.items(), inplength_o))
        # Update self's dictionary
        self.inpDic.update(tosend_o)
        self.inpList.append(tosend_o)
        self.inputCount += 1
        # Update other's dictionary
        other.inpDic.update(tosend_s)
        other.inpList.append(tosend_s)
        other.inputCount += 1
        # Updating the textual history of the communication
        updated = "\tAgent " + str(self.ID) + " received " + str(inplength_o) + " items from Agent " + str(other.ID)
        self.commHistory.append(updated)
        if (verbose):
            print(updated)
        updated = "\tAgent " + str(other.ID) + " received " + str(inplength_s) + " items from Agent " + str(self.ID)
        other.commHistory.append(updated)
        if (verbose):
            print(updated)

    def updateEnv (self, linedata, ITERATION_INPUT):
        items = int(random.uniform(0.8, 1)*ITERATION_INPUT)
        idx = np.random.choice(linedata.shape[0], items, replace=False)
        data = linedata[idx, :]
        inpDicEnv = dict(zip(idx, data.tolist() ))
        self.inpList.append(inpDicEnv)   # Appending to the list of dictionary
        self.inpDic.update(inpDicEnv)    # Updating the whole dictionary, may not be useful
        self.inputCount += 1
        updated = "Agent "+ str(self.ID)+ " added "+ str(len(data))+ " items from environment"
        print(updated)


    def boidUpdateEnv (self, linedata):  #This one is each boid gets his own data

        #Taking the n-th element from the 200 element chunk. N can be 0-199

        me = self.ID
        idx = np.array([me])
        data = linedata[idx,1:]

        val = linedata[idx,0]

        inpDicEnv = dict(zip(val, data.tolist() ))
        self.inpList.append(inpDicEnv)   # Appending to the list of dictionary
        self.inpDic.update(inpDicEnv)    # Updating the whole dictionary, may not be useful
        self.inputCount += 1
        updated = "Agent "+ str(self.ID)+ " added "+ str(len(data))+ " items from environment"
        print(updated)
        #print("\tData received from environment: ", data)
        #print("\t",self.inpDic)

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
        currInputIdx = set(list(currInput.keys()))   #Get the IDs of the current myinput
        #print("Length of passed myinput for Agent", self.ID+1, "is", len(currInputIdx))

        #This is the case when agent is training for the first time
        if self.initFlag == True:
            toTrainIdx = currInputIdx
            self.repository = self.repository.union(currInputIdx)
            self.initFlag = False
            #print("When Init Flag is True")
            #print("Initial Repository length for Agent", self.ID+1,len(self.repository))

        else:
            toTrainIdx = currInputIdx - self.repository  # Get only the IDs that are not in repository already
            self.repository = self.repository.union(toTrainIdx)  # Update repository by adding the current IDs

        #print("Final Repository length for Agent", self.ID+1,"is",len(self.repository))
        #Now building a dictionary with only the new values and returning the values
        toTrainIdx = list(toTrainIdx)
        #print("Length of returned myinput", len(toTrainIdx))
        toTrainDict = {k: currInput[k] for k in toTrainIdx} #Taking only the selected entries fo the myinput and builda new dicitonary
        return np.asarray(list(toTrainDict.values()))