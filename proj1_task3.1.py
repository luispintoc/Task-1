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

def task31(x): #add bias
	one = np.ones(len(x))
	newx = np.column_stack((x,one))
	return newx

def error_print(y_train,y_val):
	error_training = np.square(np.subtract(y_train, y_training)).mean()
	error_validation = np.square(np.subtract(y_val, y_validation)).mean()
	print('The mean-squared error on the training set is:', error_training)
	print('The mean-squared error on the validation set is:', error_validation)
	return error_training, error_validation



[x_tr, y_training] = splitData(data,0,10000,'Task3.1')
x_training = task31(x_tr)

[x_v, y_validation] = splitData(data,10000,11000,'Task3.1')
x_validation = task31(x_v)

[x_te, y_test] = splitData(data,0,1000,'Task3.1')
x_test = task31(x_te)


#Closed_form approach
time_closed_list = []

start1 = time.time()
w = []
w = closed_form(x_training, y_training)
yclosed_predicted_training = np.matmul(x_training,w)
yclosed_predicted_val = np.matmul(x_validation,w)

print('Closed form approach:')
error_print(yclosed_predicted_training, yclosed_predicted_val)

end1 = time.time()
print('Time elapsed: ',end1-start1)
closed_time = end1-start1
time_closed_list.append(closed_time)

# i = 0
# while i < 20:
# 	i += 1
# avg_time = sum(time_closed_list)/len(time_closed_list)

# print('The closed-form runtime is: ', avg_time)


#Gradient descent approach
w0 = np.random.random((len(x_training[0]),1)) #initialization between [0,1]
epsilon = 0.001
regularization = 0
beta = np.linspace(0,0.4,500)
eta0 = 0.47
start = time.process_time()
wd = []
wd = grad_des(x_training,y_training,w0,beta,eta0,epsilon,regularization) #X, Y, w0, beta, eta0, eps, r

print('Gradient descent approach:')
ygrad_predicted_train = np.matmul(x_training,wd)
ygrad_predicted_val = np.matmul(x_validation,wd)

[error_training , error_validation ] = error_print(ygrad_predicted_train,ygrad_predicted_val)


end = time.process_time()
print('Time elapsed: ',end-start)



# '''
# #Gradient descent approach
# eta0_list = []
# step_size_list = []
# time_list = []
# yv_list = []
# yt_list = []
# i = 1
# n = 1500 #number of points in the plot
# while i < n:

# 	#Parameters	
# 	w0 = np.random.random((len(x_training[0]),1)) #initialization between [0,1]
# 	epsilon = 0.001
# 	regularization = 0

# 	#uncomment this lines to generate the plots for different learning rates:
# 	'''
# 	beta = np.linspace(0, 0.1*4 ,len(x_training)) #set step-size in beta to a constant decay value
# 	a = np.linspace(0,0.1465,n)
# 	eta0 = 0.35 + a[i] #eta0 will change from [0.35 to 0.5]
# 	'''

# 	#uncomment this lines to generate plots for different speeds of decay
# 	beta = np.linspace(0,0.4,i+500) #vary beta from 0 to 0.4
# 	step_size = []
# 	step_size = (beta[1]-beta[0])/len(beta)
# 	eta0 = 0.47 #set eta0 to a constant value to test for beta


# 	start = time.process_time()


# 	wd = []
# 	wd = grad_des(x_training,y_training,w0,beta,eta0,epsilon,regularization) #X, Y, w0, beta, eta0, eps, r
# 	ygrad_predicted_train = np.matmul(x_training,wd)
# 	ygrad_predicted_val = np.matmul(x_validation,wd)

# 	[error_training , error_validation ] = error_print(ygrad_predicted_train,ygrad_predicted_val)


# 	end = time.process_time()
# 	time_list.append(end - start)

# 	#eta0_list.append(eta0) #uncomment this line for different learning rates
# 	step_size_list.append(step_size) #uncomment this line for different speeds of decay

# 	yt_list.append(error_training)
# 	yv_list.append(error_validation)

# 	i += 1

# #Plots

# #code to plot different initial learning rates
# '''
# pt.figure(1)
# pt.scatter(time_list,eta0_list,s=10,color='b')
# one2 = np.ones(len(eta0_list))*0.48
# one3 = np.ones(len(eta0_list))*0.468
# pt.plot(time_list,one2,'k-',color='k')
# pt.plot(time_list,one3,'k-',color='k')
# axes = pt.gca()
# axes.set_ylim([min(eta0_list),max(eta0_list)])
# axes.set_xlim([0,2.25])
# pt.ticklabel_format(style='sci',axis='y',scilimits=(-7,-5))
# pt.xlabel('time [s]')
# pt.ylabel('${\eta_0}$')
# pt.title('Runtime for different initial learning rates')
# #print(time_list)
# #pt.show()

# pt.figure(2)
# pt.scatter(eta0_list,yt_list,s=10,color='b')
# pt.plot(one2,yt_list,'k-',color='k')
# pt.plot(one3, yt_list,'k-',color='k')
# axes = pt.gca()
# axes.set_xlim([min(eta0_list),max(eta0_list)])
# axes.set_ylim([min(yt_list),max(yt_list)])
# pt.ticklabel_format(style='sci',axis='x',scilimits=(-7,-5))
# pt.ylabel('Error')
# pt.xlabel('${\eta_0}$')
# pt.title('Mean-squared error on the training set as a function of ${\eta_0}$')

# pt.figure(3)
# pt.scatter(eta0_list,yv_list,s=10,color='b')
# pt.plot(one2,yv_list,'k-',color='k')
# pt.plot(one3, yv_list,'k-',color='k')
# axes = pt.gca()
# axes.set_xlim([min(eta0_list),max(eta0_list)])
# axes.set_ylim([min(yv_list),max(yv_list)])
# pt.ticklabel_format(style='sci',axis='x',scilimits=(-7,-5))
# pt.xlabel('${\eta_0}$')
# pt.ylabel('Error')
# pt.title('Mean-squared error on the validation set as a function of ${\eta_0}$')
# pt.show()

# '''

# #code to plot stepsize in beta (speed of the decay)


# pt.figure(1)
# pt.scatter(time_list,step_size_list,s=15,color='k')
# one3 = np.ones(len(time_list))*0.0000003
# pt.plot(time_list,one3,'k-',color='r')
# axes = pt.gca()
# axes.set_ylim([min(step_size_list),max(step_size_list)])
# pt.ticklabel_format(style='sci',axis='y',scilimits=(-9,-7))
# pt.xlabel('time[s]')
# pt.ylabel('step size in \u03B2')
# pt.title('Runtime for different decay rates')


# pt.figure(2)
# pt.scatter(step_size_list,yt_list,s=15,color='b')
# #one2 = np.ones(len(time_list))*0.0000003
# pt.plot(one3,yt_list,'k-',color='r')
# axes = pt.gca()
# axes.set_xlim([min(step_size_list),max(step_size_list)])
# axes.set_ylim([min(yt_list),max(yt_list)])
# pt.ticklabel_format(style='sci',axis='x',scilimits=(-9,-7))
# pt.ylabel('Error')
# pt.xlabel('Step Size')
# pt.title('Mean-squared error on the training set as a function of \u03B2 step size')

# pt.figure(3)
# pt.scatter(step_size_list,yv_list,s=15,color='b')
# #one2 = np.ones(len(time_list))*0.0000003
# pt.plot(one3,yv_list,'k-',color='r')
# axes = pt.gca()
# axes.set_xlim([min(step_size_list),max(step_size_list)])
# axes.set_ylim([min(yv_list),max(yv_list)])
# pt.ticklabel_format(style='sci',axis='x',scilimits=(-9,-7))
# pt.xlabel('Step_size')
# pt.ylabel('Error')
# pt.title('Mean-squared error on the validation set as a function of \u03B2 step size')
# pt.show()
