import cvxpy as cp
import math
import networkx as nx
import random
import numpy as np

def create(n):
    np.random.seed(0)

    G = nx.connected_watts_strogatz_graph(n,4,0.5)
    domain = ['A','B','C','D','E']

    for n in G.nodes():
        G.node[n]['domain'] = domain[:int(math.ceil(random.uniform(0,len(domain))))]
        G.node[n]['unary_potential'] = np.random.randn(len(G.node[n]['domain']))
        G.node[n]['mu'] = cp.Variable(len(G.node[n]['domain']))

    for a,b in G.edges():
        G[a][b]['pairwise_potential'] = np.random.randn(len(G.node[a]['domain']), len(G.node[b]['domain']))
        G[a][b]['mu'] = cp.Variable(len(G.node[a]['domain']), len(G.node[b]['domain']))

    dot = lambda a,b : cp.sum_entries(cp.mul_elemwise(a,b))

    f = (sum([dot(G.node[n]['unary_potential'],G.node[n]['mu']) for n in G.nodes()]) +
         sum([dot(G[a][b]['pairwise_potential'],G[a][b]['mu']) for a,b in G.edges()]))
    C = ([G[a][b]['mu']*np.ones(len(G.node[b]['domain'])) == G.node[a]['mu'] for a,b in G.edges()] +
         [G[a][b]['mu'].T*np.ones(len(G.node[a]['domain'])) == G.node[b]['mu'] for a,b in G.edges()] +
         [G[a][b]['mu'] >= 0 for a,b in G.edges()] +
         [cp.sum_entries(G.node[n]['mu'])==1 for n in G.nodes()])

    return cp.Problem(cp.Minimize(f), C)
