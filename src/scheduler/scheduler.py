
from src.scheduler.topologicalSort.topoSort import topoGraph
from src.scheduler.topologicalSort.adjacencyList import adjacencyList


def scheduler(nodeDict:dict,nodeList:list):
    # Generate adjacency List of the operator
    adj = adjacencyList()
    DAG = adj.findDAG(nodeDict, nodeList)
    adjlist = adj.initGraph(DAG, len(nodeList))
    
    # Scheduler
    topo = topoGraph(len(nodeList))
    for i in range(len(adjlist)):
        for j in adjlist[i]:
            topo.addEdge(i,j)
    return topo.topologicalSort()