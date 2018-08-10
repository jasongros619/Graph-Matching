from Synthetic_Graph_Generation import *
#g1,g2,perm = synthetic_Graph_Generation(n_in,n_out,variance)
#W = affinity_Matrix_From_Graphs(graph1,graph2,distance=RBF_Distance)

from Multiplicative_Update_Algorithms import *
#Run_Full_NOGM_Alg(W,n_updates,X_init=None)
#Run_Full_MPGM_Alg(W,n_updates,X_init=None)



#Synthetic graph matching
#Fixed N, Variance, updates
def Test_1_NOGM(N,variance,n_updates,tries=100):
    tot_score = 0
    for t in range(tries):
        g1,g2,perm_mat = synthetic_Graph_Generation(N,0,variance)
        W = affinity_Matrix_From_Graphs(g1,g2)
        
        proposed = Run_Full_NOGM_Alg(W,n_updates)
        proposed2= Run_Full_NOGM_Alg(W,n_updates*4)
        proposed4= Run_Full_NOGM_Alg(W,n_updates*16)
        proposed8= Run_Full_NOGM_Alg(W,n_updates*64)
        proposed6= Run_Full_NOGM_Alg(W,n_updates*128)
        
        


        print(proposed.T)
        print("")
        print(proposed2.T)
        print("")
        print(proposed4.T)
        print("")
        print(proposed8.T)
        print("")
        print(proposed6.T)
        print("")
        print(proposed.T+2*proposed2.T+4*proposed4.T+8*proposed8.T+16*proposed6.T)
        print(perm_mat)
        
        score = sum(sum(proposed.T*perm_mat))/N
        tot_score += score
    return tot_score / tries

print(Test_1_NOGM(14,0,200,1))

def Test_1_MPGM(N,variance,n_updates,tries=100):
    tot_score = 0
    for t in range(tries):
        g1,g2,perm_mat = synthetic_Graph_Generation(N,0,variance)
        W = affinity_Matrix_From_Graphs(g1,g2)        
        proposed = Run_Full_MPGM_Alg(W,n_updates)
        score = sum(sum(proposed.T*perm_mat))/N
        tot_score += score
    return tot_score / tries



