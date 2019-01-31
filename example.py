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
import time

# It a list of data points, where each datapoint is a dictionary with the following attributes:
# popularity_score : a popularity score for this comment (based on the number of upvotes) (type: float)
# children : the number of replies to this comment (type: int)
# text : the text of this comment (type: string)
# controversiality : a score for how "controversial" this comment is (automatically computed by Reddit)
# is_root : if True, then this comment is a direct reply to a post; if False, this is a direct reply to another comment 

with open("proj1_data.json") as fp:
    data = json.load(fp)


#for Task 3.1
x_training = []
x_validation = []
x_test = []
time_list = []
time1_list = []

def task31(x): #add bias
	one = np.ones(len(x))
	newx = np.column_stack((x,one))
	return newx

def error_print(y_train,y_val):
	error_training = np.square(np.subtract(y_train, y_training)).mean()
	error_validation = np.square(np.subtract(y_val, y_validation)).mean()
	#print('The mean-squared error on the training set is:', error_training)
	#print('The mean-squared error on the validation set is:', error_validation)
	return error_training, error_validation



[x_tr, y_training] = splitData(data,0,10000,'Task3.1')
x_training = task31(x_tr)

[x_v, y_validation] = splitData(data,10000,11000,'Task3.1')
x_validation = task31(x_v)

[x_te, y_test] = splitData(data,0,1000,'Task3.1')
x_test = task31(x_te)

i = 0
while i<1000:

	#Closed_form approach
	start1 = time.time()
	w = []
	w = closed_form(x_training, y_training)
	yclosed_predicted_training = np.matmul(x_training,w)
	yclosed_predicted_val = np.matmul(x_validation,w)

	#print('Task 3.1: Linear regression using closed-form approach:')
	[error_training,error_validation]=error_print(yclosed_predicted_training, yclosed_predicted_val)
	end1 = time.time()
	time1_list.append(end1-start1)

	time.sleep(0.01)

	#Gradient descent approach
	w0 = np.random.random((len(x_training[0]),1)) #initialization between [0,1]
	epsilon = 0.001
	regularization = 39
	beta = np.linspace(0,0.4,500)
	eta0 = 0.47
	start = time.time()
	wd = []
	wd = grad_des(x_training,y_training,w0,beta,eta0,epsilon,regularization) #X, Y, w0, beta, eta0, eps, r

	#print('Task 3.1: Linear regression using gradient descent approach:')
	ygrad_predicted_train = np.matmul(x_training,wd)
	ygrad_predicted_val = np.matmul(x_validation,wd)

	[error_training , error_validation ] = error_print(ygrad_predicted_train,ygrad_predicted_val)


	end = time.time()
	time_list.append(end-start)

	i+=1

runs = np.linspace(0,1000,1000)
pt.figure(1)
pt.scatter(runs,time_list,s=15,color='k')
pt.scatter(runs,time1_list,s=15,color='b')
one3 = np.ones(len(runs))
avg1 = sum(time_list)/len(time_list)
avg2 = sum(time1_list)/len(time_list)
print('Grad:',avg1, 'Closed:',avg2)
avg1 = one3*avg1
avg2 = one3 * avg2
pt.plot(runs,avg1,'k-',color='r')
pt.plot(runs,avg2,'k-',color='r')
axes = pt.gca()
#pt.legend('Avg grad desc','Avg closed form')
axes.set_ylim([0,0.020])
pt.ylabel('time[s]')
pt.xlabel('Runs')
pt.title('Runtime comparison')
pt.show()
