'''
Mini-Project1 - COMP 551 - Winter 2019
Mahyar Bayran
Luis Pinto
Rebecca Salganik
'''

import json # we need to use the JSON package to load the data, since the data is stored in JSON format
import numpy as np
import matplotlib.pyplot as pt
from proj1_task1 import splitData
from proj1_task2 import closed_form
from proj1_task2 import grad_des
import csv
import time


with open("proj1_data.json") as fp:
    data = json.load(fp)

def toFloat(x):

    X = np.zeros( (len(x), len(x[0])) )
    for i in range(0, len(x)):
        for j in range(0, len(x[0])):
            X[i][j] = float(x[i][j])
    return X

def bias(x):
    one = np.ones(len(x))
    newx = np.column_stack((x,one))
    return newx

def error_print(y_train,y_val,y_tes):
    error_train = np.square(np.subtract(y_train, y_training)).mean()
    error_validation = np.square(np.subtract(y_val, y_validation)).mean()
    error_test = np.square(np.subtract(y_tes, y_test)).mean()
    print('The MSE on the training set is:', error_train)
    print('The MSE on the validation set is:', error_validation)
    print('The MSE on the test set is:', error_test)
    return error_train,error_validation,error_test

def Select(x, num):
    y = np.zeros( (len(x), num) )
    for i in range(0, len(x)):
        for j in range(0, num):
            y[i][j] = x[i][j]

    return y

# number of top bigram counts to be used as features (max 30)
B = 24
# number of top words to be used as features (max 160)
C = 7

print('Task3.3: Closed-form method using the 3 simple features + top 24 bigrams + 7 top words + length of the comment')
print('Takes about 10 min to run the code, please be patient')

start = time.time()

#for Task 3.2
[x1_tr, bigrams_tr, topWords_tr, y_training] = splitData(data,0,100,'Task3.3')
np.savetxt("bigrams_tr.csv", bigrams_tr, delimiter=",")
x1_tr = np.column_stack((x1_tr, Select( bigrams_tr, B)))
x1_tr = np.column_stack((x1_tr, Select(topWords_tr, C)))
x1_tr = bias(x1_tr)

[x1_v, bigrams_v, topWords_v, y_validation] = splitData(data,100,110,'Task3.3')
np.savetxt("bigrams_v.csv", bigrams_v, delimiter=",")
x1_v = np.column_stack((x1_v, Select(bigrams_v, B)))
x1_v = np.column_stack((x1_v, Select(topWords_v, C)))
x1_v = bias(x1_v)

[x1_te, bigrams_test, topWords_test, y_test] = splitData(data,110,120,'Task3.3')
np.savetxt("bigrams_test.csv", bigrams_test, delimiter=",")
x1_te = np.column_stack((x1_te, Select(bigrams_test, B)))
x1_te = np.column_stack((x1_te, Select(topWords_test, C)))
x1_te = bias(x1_te)


def task33(xt,xv,xtest,y):
    w = closed_form(xt,y)
    y_predicted_train = np.matmul(xt,w)
    y_predicted_val = np.matmul(xv,w)
    y_predicted_test = np.matmul(xtest,w)
    error_print(y_predicted_train,y_predicted_val,y_predicted_test)

end = time.time()
elapsed = end - start
print('Errors:')
task33(x1_tr, x1_v, x1_te, y_training)
print('Time elapsed: ', (end - start))
print('This is our best-performing model')
