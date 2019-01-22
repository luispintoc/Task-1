import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
from sklearn import linear_model

original_training_data = pd.read_csv('C:/Users/luizi/Documents/COMP 551/Least Square Method/train.csv')
training_data = original_training_data.dropna()
original_test_data = pd.read_csv('C:/Users/luizi/Documents/COMP 551/Least Square Method/test.csv')
test_data = original_test_data.dropna()
#print(test_data.head(10))

x = training_data[['x']].as_matrix()
y = training_data[['y']].as_matrix()


xtest = test_data[['x']].as_matrix()
ytest = test_data[['y']].as_matrix()

one = np.ones(len(x))

newx = np.column_stack((x,one))



lm = linear_model.LinearRegression()
lm.fit(x,y)

pt.figure(3)
pr = lm.predict(xtest)
print('The correlation between outputs with sklearn regression method is:',math.sqrt(lm.score(ytest,pr)))
pt.scatter(ytest,pr,s=5,color='c')
pt.title('Linear Regression SkLearn')



#pt.show()



def closed_form(X, Y):

    g = np.matmul(np.linalg.inv( np.matmul(X.T, X) ) ,np.matmul(X.T, Y))    
    
    return g

def grad_des(X, Y, w0, beta, eta0, eps):

    # X should have the bias term
    # w0 = np.random.random((len(X[0]),1))
    d = 1e10 # sth large
    i = 0
    w_old = w0
    
    while ( ~( (d<eps)|(i>len(beta)-1) ) ):
        
        alfa = eta0/(1+beta[i])
        w = w_old - 2*alfa*( np.matmul(X.T, np.matmul(X, w_old) - Y) )

        d = np.linalg.norm( w - w_old )
        w_old = w

        i += 1

    return w

g = closed_form(newx,y)
ypredicted2 = g[[0]]*xtest + g[[1]]
print('The line equation is: ',g[[0]],'x + ',g[[1]])
print('The correlation between outputs with closed-form method is:', math.sqrt(lm2.score(ytest,ypredicted2)))

w0 = np.random.random((len(x[0]),1))
beta = list(range(0,300))
#beta = [x / 300 for x in beta]
beta = [0 for x in beta]

#newx = np.squeeze(np.asarray(newx))
#y = np.squeeze(np.asarray)
j = grad_des(np.asarray(newx),np.asarray(y),w0,beta, 0.01,0.0001)
ypredicted3 = j[[0]]*xtest + j[[1]]
print('The line equation is: ',j[[0]],'x + ',j[[1]])
print('The correlation between outputs with closed-form method is:', math.sqrt(lm2.score(ytest,ypredicted3)))

