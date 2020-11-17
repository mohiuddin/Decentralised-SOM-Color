

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