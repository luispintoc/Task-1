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
import time

with open("proj1_data.json") as fp:
    data = json.load(fp)


def bias(x):
	one = np.ones(len(x))
	newx = np.column_stack((x,one))
	return newx

def error_print(y_train,y_val,):
	error_training = np.square(np.subtract(y_train,y_training)).mean()
	error_validation = np.square(np.subtract(y_val, y_validation)).mean()
	print('The mean-squared error on the training set is:', error_training)
	print('The mean-squared error on the validation set is:', error_validation)


#for Task 3.2
[x1_t,x2_t,x3_t,y_training] = splitData(data,0,10000,'Task3.2')
x1training = bias(x1_t)
x2training = bias(x2_t)
x3training = bias(x3_t)

[x1_v, x2_v, x3_v, y_validation] = splitData(data,10000,11000,'Task3.2')
x1validation = bias(x1_v)
x2validation = bias(x2_v)
x3validation = bias(x3_v)


def task32(xt,xv,y):
	w = closed_form(xt,y)
	y_predicted_train = np.matmul(xt,w)
	y_predicted_val = np.matmul(xv,w)
	error_print(y_predicted_train,y_predicted_val)

print('Task 3.2: Linear regression using closed-form approach')
start = time.time()
print('Errors for set with no text features:')
task32(x1training,x1validation,y_training)
end = time.time()
print('Time elapsed using no text features: ', end-start)
start = time.time()
print('Errors for set with top 60 words:')
task32(x2training,x2validation,y_training)
end = time.time()
print('Time elapsed using top 60 words: ', end-start)
start = time.time()
print('Errors for set with top 160 words:')
task32(x3training,x3validation,y_training)
end = time.time()
print('Time elapsed using top 160 words: ', end-start)
