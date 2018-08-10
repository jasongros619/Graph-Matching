import numpy as np
import scipy.optimize

#used to go from approximate answers to permutation matrices
def Get_Perm_From_Approximate_Matrix(X_mat):
    cost = -X_mat
    x,y = scipy.optimize.linear_sum_assignment( cost )
    perm = np.zeros(X_mat.shape)
    for i in range(X_mat.shape[0]):
        perm[ x[i],y[i] ] = 1
    return perm

#Nonnegative Orthogonal Graph Matching
def NOGM_update_iteration(W,X_mat):
    X_vec = X_mat.reshape(-1,1)
    
    K = np.reshape(np.dot(W,X_vec),X_mat.shape)

    tmp = np.dot(K,X_mat.T)
    delta = 0.5*( tmp + tmp.T )

    X_mat *= (K / np.dot(delta,X_mat))**0.5

def Run_Full_NOGM_Alg(W,n_updates,X_init=None):
    N = int(W.shape[0]**0.5)
    if X_init is None:
        X_init = np.ones( (N,N) )
        X_init[0,0] = 0
    
    X_mat = X_init

    for i in range(n_updates):
        NOGM_update_iteration(W,X_mat)
    print( (100*X_mat).astype(np.int32)/100 )
    proposed = Get_Perm_From_Approximate_Matrix(X_mat)

    return proposed

#Multiplicative Update Graph Matching
def MPGM_update_iteration(W,X_mat):
    X_vec = X_mat.reshape(-1,1)
    N = X_mat.shape[0]

    K = np.dot(W,X_vec).reshape(X_mat.shape)

    gamma = 2* np.linalg.inv(np.identity(N)-np.dot(X_mat.T,X_mat))
    tmp = np.diag( np.dot(K.T,X_mat) - np.dot(X_mat.T,np.diag(np.dot(K,X_mat.T))))
    gamma = np.dot(gamma,tmp)

    delta = 2*np.diag( np.dot(K,X_mat.T) )-np.dot(X_mat,gamma)

    gamma_plus = 0.5 * (np.abs(gamma) + gamma)
    gamma_minus= 0.5 * (np.abs(gamma) - gamma)
    delta_plus = 0.5 * (np.abs(delta) + delta)
    delta_minus= 0.5 * (np.abs(delta) - delta)

    #for easier calculations ahead [gamma_0 | gamma_1 | ... ] & [lambda_0 | lambda_1 | ... ].T
    gamma_plus_matrix = np.concatenate( [gamma_plus.reshape(1,-1)  for i in range(N)], axis=0)
    gamma_minus_matrix= np.concatenate( [gamma_minus.reshape(1,-1) for i in range(N)], axis=0)
    delta_plus_matrix = np.concatenate( [delta_plus.reshape(-1,1)  for i in range(N)], axis=1)
    delta_minus_matrix= np.concatenate( [delta_minus.reshape(-1,1) for i in range(N)], axis=1)

    numerator =  (2 * K + gamma_minus_matrix + delta_minus_matrix)
    denominator = gamma_plus_matrix + delta_plus_matrix

    ratio = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    X_mat *= (ratio)**0.5

def Run_Full_MPGM_Alg(W,n_updates,X_init=None):
    N = int(W.shape[0]**0.5)
    if X_init is None:
        X_init = np.ones( (N,N) )
        X_init[0,0] = 0
    
    X_mat = X_init

    for i in range(n_updates):
        MPGM_update_iteration(W,X_mat)
    proposed = Get_Perm_From_Approximate_Matrix(X_mat)

    return proposed
