import numpy as np
import itertools
from utils import reachable


def skeleton(indepTest, labels, m_max, alpha=0.05, priorAdj=None, **kwargs):
    sepset = [[None for i in range(len(labels))] for i in range(len(labels))]

    # form complete undirected graph, true if edge i--j needs to be investigated
    G = [[True for i in range(len(labels))] for i in range(len(labels))]

    for i in range(len(labels)): G[i][i] = False

    # done flag
    done = False

    ord = 0
    n_edgetests = {}

    while done != True and any(G) and ord <= m_max:
        ord1 = ord + 1
        n_edgetests[ord1] = 0
        done = True
        G1 = G.copy()

        ind = [(i, j)
               for i in range(len(G))
               for j in range(len(G[i]))
               if G[i][j] == True
               ]
        for x, y in ind:
            if priorAdj is not None:
                if priorAdj[x,y]==1 or priorAdj[y,x]==1:
                    continue

            if G[y][x] == True:
                nbrs = [i for i in range(len(G1)) if G1[x][i] == True and i != y]
                if len(nbrs) >= ord:
                    if len(nbrs) > ord:
                        done = False

                    for nbrs_S in set(itertools.combinations(nbrs, ord)):
                        n_edgetests[ord1] = n_edgetests[ord1] + 1
                        pval = indepTest.fit(x, y, list(nbrs_S), **kwargs)
                        if pval >= alpha:
                            G[x][y] = G[y][x] = False
                            sepset[x][y] = set(nbrs_S)
                            break
        ord += 1

    return {'sk': np.array(G),'sepset': sepset,}


def pruning(indepTest, dag, m_max, alpha=0.05, priorAdj=None, **kwargs):

    for r in range(1, m_max):
        dag1 = dag.copy()
        edges = np.where(dag == 1)

        for k in range(len(edges[0])):
            xi, xj = edges[0][k], edges[1][k]

            if priorAdj is not None:
                if priorAdj[xi,xj] == 1 or priorAdj[xj,xi] ==1:
                    continue
                if priorAdj[xi,xj] == -1:
                    dag1[xi, xj] = 0
                    continue

            ifdelete = dag.copy()
            ifdelete[xi, xj] = 0

            considerz = []
            for parent in list(np.where(ifdelete[:, xi] == 1)[0]):
                if reachable(ifdelete, parent, xj): considerz.append(parent)
            for parent in list(np.where(ifdelete[:, xj] == 1)[0]):
                if reachable(ifdelete, parent, xi): considerz.append(parent)

            considerz = list(set(considerz))

            if len(considerz) > r:
                if len(considerz) == 1:
                    z = considerz[0]
                    pvalue = indepTest.fit(xi, xj, z, **kwargs)
                    if pvalue > alpha:
                        dag1[xi, xj] = 0
                        continue
                else:
                    for nbrs_z in set(itertools.combinations(considerz, r)):
                        pvalue = indepTest.fit(xi, xj, list(nbrs_z), **kwargs)
                        if pvalue > alpha:
                            dag1[xi, xj] = 0
                            break
        dag = dag1.copy()
    return dag
