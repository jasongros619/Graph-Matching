import numpy as np

#Note: Graph is symetric
def generate_Synthetic_Graph_1(N):
    graph = np.random.uniform(0,1,(N,N))

    #make it symmetric
    for i in range(0,N):
        for j in range(i,N):
            graph[i,j] = graph[j,i]
    return graph

def generate_Rand_Perm_Mat(N):
    vals = np.random.permutation(N)
    vals = [i for i in range(1,N)]+[0]
    
    output = np.zeros((N,N))
    for i,v in enumerate(vals):
        output[i,v] = 1
    return output

def generate_Synthetic_Graph_2(n_in,n_out,graph_1,perm_mat,variance):
    N = graph_1.shape[0]
    assert (n_in + n_out == N),"n_in("+str(n_in)+" and n_out("+str(n_out)+") do not match size of matrix("+str(graph_1.shape)+")"

    #compute inlier edges and permute
    graph2 = np.array(graph_1[:n_in,:n_in])
    noise = np.random.normal(0,variance**2,(n_in,n_in))
    for i in range(0,n_in):
        for j in range(i,n_in):
            noise[i,j]=noise[j,i]
    graph2 += noise
    graph2 = np.dot(perm_mat, graph2)

    #add outlier
    output = generate_Synthetic_Graph_1(N)
    output[:n_in,:n_in] = graph2

    return output


def synthetic_Graph_Generation(n_in,n_out,variance):
    graph1 = generate_Synthetic_Graph_1(n_in + n_out)
    perm_mat = generate_Rand_Perm_Mat(n_in)
    graph2 = generate_Synthetic_Graph_2(n_in,n_out,graph1,perm_mat,variance)

    return graph1,graph2,perm_mat

# 0.15 for synthetic graph matching
def RBF_Distance(a,b,scale=0.015):
    return np.exp( -(a-b)**2/scale )


def affinity_Matrix_From_Graphs(graph1,graph2,distance=RBF_Distance):
    N = graph1.shape[0]
    N2 = N**2

    affinity = np.zeros( (N2,N2) )

    #edges  Wij,kl = S(e_ik,e'_jl)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    if True:
                        affinity[i*N+j,k*N+l] = distance(graph1[i,k],graph2[j,l])

    return affinity
