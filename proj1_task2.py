'''
Mini-Project1 - COMP 551 - Winter 2019
Mahyar Bayran
Luis Pinto
Rebecca Salganik
'''

import numpy as np

def closed_form(X, Y):

    x1 = np.matmul(X.T, X)
    x2 = np.matmul(X.T, Y)
    w = np.matmul(np.linalg.inv(x1) , x2)
    
    return w


def grad_des(X, Y, w0, beta, eta0, eps, r):

    # X should have the bias term
    # w0 = np.random.random((len(X[0]),1))
    d = 1e10 # sth large
    i = 0
    w_old = w0
    m = len(Y)
    
    while ( ~( (d<eps)|(i>len(beta)-1) ) ):
        
        alfa = eta0/(1+beta[i])
        w = w_old - 2*alfa * (1.0/m) * (np.matmul(X.T, np.matmul(X, w_old)) - np.matmul(X.T,Y) - r*w_old)
        d = np.linalg.norm( w - w_old )
        w_old = w        
        i += 1

    return w
