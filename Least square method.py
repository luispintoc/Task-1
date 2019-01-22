import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
from sklearn import linear_model

original_training_data = pd.read_csv('train.csv')
training_data = original_training_data.dropna()
original_test_data = pd.read_csv('test.csv')
test_data = original_test_data.dropna()
#print(test_data.head(10))

x = training_data[['x']].as_matrix()
y = training_data[['y']].as_matrix()


xtest = test_data[['x']].as_matrix()
ytest = test_data[['y']].as_matrix()

one = np.ones(len(x))
newx = np.column_stack((x, one))



lm = linear_model.LinearRegression()
lm.fit(x,y)

pt.figure(3)
pr = lm.predict(xtest)
print('The correlation between outputs with sklearn regression method is:',math.sqrt(lm.score(ytest,pr)))
pt.scatter(ytest,pr,s=5,color='c')
pt.title('Linear Regression SkLearn')



#pt.show()



def closed_form(X, Y, r):
    # r : regularization parameter
    n = len(X[0])
    g = np.matmul(np.linalg.inv( np.matmul(X.T, X) + r*np.identity(n) ) ,np.matmul(X.T, Y))    
    
    return g

def grad_des(X, Y, w0, beta, eta0, eps, r):

    # X should have the bias term
    # w0 = np.random.random((len(X[0]),1))
    d = 1e10 # sth large
    i = 0
    w_old = w0
    m = len(Y)
    
    while ( ~( (d<eps)|(i>len(beta)-1) ) ):
        
        alfa = eta0/(1+beta[i])
        w = w_old - alfa * (1.0/m) * (np.matmul(X.T, np.matmul(X, w_old) - Y) - r*w_old)
        d = np.linalg.norm( w - w_old )
        w_old = w

        #print(w_old)
        
        i += 1

    return w

r = 0

g = closed_form(newx, y, r)
ypredicted2 = g[[0]]*xtest + g[[1]]
print('The line equation is: ',g[[0]],'x + ',g[[1]])
print('The correlation between outputs with closed-form method is:', math.sqrt(lm.score(ytest,ypredicted2)))


w0 = np.random.random((len(newx[0]),1))
beta = list(range(0,500))
#beta = [x / 300 for x in beta]
beta = [0 for x in beta]

j = grad_des(newx, y, w0, beta, 0.0003, 0.0001, r)
print(j)
ypredicted3 = j[[0]]*xtest + j[[1]]
print('The line equation is: ',j[[0]],'x + ',j[[1]])
print('The correlation between outputs with closed-form method is:', math.sqrt(lm.score(ytest,ypredicted3)))

