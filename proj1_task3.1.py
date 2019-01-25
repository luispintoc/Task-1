import json # we need to use the JSON package to load the data, since the data is stored in JSON format
import numpy as np
import matplotlib.pyplot as pt
from proj1_task1 import splitData
from proj1_task2 import closed_form
from proj1_task2 import grad_des

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

def error_print(y_val,y_tes):
	error_validation = np.square(np.subtract(y_val, y_validation)).mean()
	error_test = np.square(np.subtract(y_tes, y_test)).mean()
	print('The mean-squared error on the validation set is:', error_validation)
	print('The mean-squared error on the test set is:', error_test)


[x_tr, y_training] = splitData(data,0,10000,'Task3.1')
x_training = task31(x_tr)

[x_v, y_validation] = splitData(data,10000,11000,'Task3.1')
x_validation = task31(x_v)

[x_te, y_test] = splitData(data,11000,12000,'Task3.1')
x_test = task31(x_te)


'''

#Closed_form approach
w = []
w = closed_form(x_training, y_training)
yclosed_predicted_val = np.matmul(x_validation,w)
yclosed_predicted_test = np.matmul(x_test,w)

error_print(yclosed_predicted_val, yclosed_predicted_test)

'''

#Gradient descent approach
wd = []
w0 = np.random.random((len(x_training[0]),1))
beta = list(range(0,len(x_training)))
wd = grad_des(x_training,y_training,w0,beta,0.00001,0.000001,0) #X, Y, w0, beta, eta0, eps, r

ygrad_predicted_val = np.matmul(x_validation,wd)
ygrad_predicted_test = np.matmul(x_test,wd)

error_print(ygrad_predicted_val,ygrad_predicted_test)
