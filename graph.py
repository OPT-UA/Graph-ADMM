import networkx as nx


##### ADDITIONAL FUNCTIONS: SUBSETS CALCULATOR #####

# Auxiliary function defined recursively to be implemented in "subsets"
def calcSubset(A, res, subset, index):
    res.append(subset[:])
    for i in range(index, len(A)):
        subset.append(A[i])
        calcSubset(A, res, subset, i + 1)
        subset.pop()


# Computes all subsets of a given list A
def subsets(A):
    subset = []
    res = []
    index = 0
    calcSubset(A, res, subset, index)
    return res


##### TYPES OF ALGORITHMIC GRAPHS #####

# Complete graph
def complete(N):
    G = nx.complete_graph(N).to_directed()
    G.remove_edges_from([(i, j) for (i, j) in G.edges if i > j])
    return G


# Sequential Graph
def sequential(N):
    G = nx.path_graph(N).to_directed()
    G.remove_edges_from([(i, j) for (i, j) in G.edges if i > j])
    return G


# Parallel Up Graph
def parallel_up(N):
    G = nx.star_graph(N - 1).to_directed()
    G.remove_edges_from([(i, j) for (i, j) in G.edges if i > j])
    return G


# Parallel Down Graph
def parallel_down(N):
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    G.add_edges_from((i, N - 1) for i in range(N - 1))
    return G


# Biparallel Graph
def biparallel(N):
    G = parallel_up(N)
    G.add_edges_from(parallel_down(N).edges())
    return G


# Ring Graph
def ring(N):
    G = sequential(N)
    G.add_edge(0, N - 1)
    return G


##### ALGORITHMIC GRAPH CALCULATOR #####

# From a (directed) graph X, collects all connected subgraphs of X ---if X=complete(n), it gives us all algorithmic graphs
def connected(X):
    Y = []
    for S in subsets(list(X.edges)):
        H = X.copy()
        H.clear_edges()
        H.add_edges_from(S)
        if nx.is_connected(H.to_undirected()): Y.append(H)
    return Y


# From a (directed) graph X, collects all subgraphs of X s.t. every node has a unique predecessor ---in particular, they are trees
def tree(X):
    Y = []
    for H in connected(X):
        if [len(H.pred[i]) for i in list(H)[1:]] == (H.order() - 1) * [1]: Y.append(H)
    return Y


# Generates a triple with all possible combinations of algorithmic graphs of order n+1
def combination_graph(n):
    X = []
    for G in connected(complete(n)):
        for Gp in connected(G):
            for Gpp in tree(G):
                X.append((G, Gp, Gpp))
    return X


# Check if a graph is complete
def is_complete(G):
    n = G.order()
    return n * (n - 1) / 2 == G.size()


# Given a tree X, returns the unique node j such that (j,i) in G.edge ---defined to simplify code-notation
def T(i, X):
    return list(X.predecessors(i))[0] if i > 0 else -1


##### GRAPH MATRICES ######

def Incidence(X):
    return -nx.incidence_matrix(X, oriented=True).toarray()


def Adjacency(X):
    return (nx.adjacency_matrix(X).toarray())


def Degree(X):
    from numpy import array, diag
    return diag(array([d for _, d in X.degree()]))


def Laplacian(X):
    return Degree(X) - Adjacency(X) - Adjacency(X).T


def Delta(X):
    return Degree(X) - 2 * Adjacency(X).T


def onto(X, sparse=True):
    from numpy import sqrt, matmul, zeros, diag
    from numpy.linalg import eigh
    if nx.is_tree(X) and sparse:
        return Incidence(X)
    elif is_complete(X) and sparse:
        N = X.order()
        Z = zeros((N, N - 1))
        for i in range(N - 1):
            Z[i, i] = sqrt((N - 1 - i) / (N - i))
            Z[(i + 1):, i] = -sqrt(1 / ((N - 1 - i) * (N - i)))
        return sqrt(N) * Z
    else:
        N = X.order()
        vals, vecs = eigh(Laplacian(X))
        idx = vals.argsort()[::-1]
        pos_vals = vals[idx[:N - 1]]
        pos_vecs = vecs[:, idx[:N - 1]]
        return pos_vecs @ diag(sqrt(pos_vals))


def degree_balance(X):
    from numpy import array
    return array([X.out_degree(n) - X.in_degree(n) for n in X.nodes()])


def alpha(X, Y):
    from numpy.linalg import pinv
    return pinv(onto(Y)) @ degree_balance(X)