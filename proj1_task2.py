'''
Mini-Project1 - COMP 551 - Winter 2019

Mahyar Bayran
Luis Pinto
Rebecca Salganik
'''

import numpy as np

# X : N*(m+1)
# Y : N*1

def closed_form(X, Y):

    w = np.matmul(np.linalg.inv( np.matmul(X.T, X) ) ,np.matmul(X.T, Y))    
    
    return w


def grad_des(X, Y, w0, beta, eta0, eps):

    # X should have the bias term
    
    d = 1e10 # sth large
    i = 0
    
    while ( ~( (d<eps)|(i>len(beta)-1) ) ):
        
        alfa = eta0/(1+beta[i])
        w = w_old - 2*alfa*( np.matmul(X.T, np.matmul(X, w_old) - Y) )

        d = np.linalg.norm( w - w_old )
        w_old = w

        i += 1

    return w

'''
# for testing
X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6]] , dtype=np.float64)
Y = np.array([[1.5], [2], [3], [4], [3.5], [3.8]], dtype=np.float64)

w = closed_form(X, Y)
print(w)
'''
