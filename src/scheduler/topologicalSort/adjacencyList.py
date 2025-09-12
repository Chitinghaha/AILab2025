class adjacencyList():
    def __init__(self) -> None: pass
    def findDAG(self, nodeDict:dict, nodeList:list) -> list:
        array = []
        for curNode in nodeDict:
            for nxtNode in nodeDict:
                for nextInput in nodeDict[nxtNode]['input']:
                    if nextInput == nodeDict[curNode]['output'][0]:
                        array.append((nodeList.index(curNode),nodeList.index(nxtNode)))
        return array
    def initGraph(self, edgeList:list, nodeLen:list) -> list:
        adjList = [0] * nodeLen
        for i in range(0, len(adjList)):
            adjList[i] = []
        for input, output in edgeList:
            adjList[input].append(output)
        return adjList
    
    def showStructure(self, adjList:list):
        for i in range(len(adjList)):
            print(f"{i}",end="")
            for leaf in adjList[i]:
                print(f" ->  {leaf}",end="")
            print()