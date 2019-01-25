import json # we need to use the JSON package to load the data, since the data is stored in JSON format
import numpy as np
import matplotlib.pyplot as pt
from proj1_task1 import splitData
from proj1_task2 import closed_form

# It a list of data points, where each datapoint is a dictionary with the following attributes:
# popularity_score : a popularity score for this comment (based on the number of upvotes) (type: float)
# children : the number of replies to this comment (type: int)
# text : the text of this comment (type: string)
# controversiality : a score for how "controversial" this comment is (automatically computed by Reddit)
# is_root : if True, then this comment is a direct reply to a post; if False, this is a direct reply to another comment 

with open("proj1_data.json") as fp:
    data = json.load(fp)


def bias(x):
	one = np.ones(len(x))
	newx = np.column_stack((x,one))
	return newx

def error_print(y_val,y_tes):
	error_validation = np.square(np.subtract(y_val, y_validation)).mean()
	error_test = np.square(np.subtract(y_tes, y_test)).mean()
	print('The mean-squared error on the validation set is:', error_validation)
	print('The mean-squared error on the test set is:', error_test)


#for Task 3.2
[x1_t,x2_t,x3_t,y_training] = splitData(data,0,10000,'Task3.2')
x1training = bias(x1_t)
x2training = bias(x2_t)
x3training = bias(x3_t)

[x1_v, x2_v, x3_v, y_validation] = splitData(data,10000,11000,'Task3.2')
x1validation = bias(x1_v)
x2validation = bias(x2_v)
x3validation = bias(x3_v)

[x1_te, x2_te, x3_te, y_test] = splitData(data,11000,12000,'Task3.2')
x1test = bias(x1_te)
x2test = bias(x2_te)
x3test = bias(x3_te)

def task32(xt,xv,xtest,y):
	w = closed_form(xt,y)
	y_predicted_val = np.matmul(xv,w)
	y_predicted_test = np.matmul(xtest,w)
	error_print(y_predicted_val,y_predicted_test)


print('Errors for set with no text features:')
task32(x1training,x1validation,x1test,y_training)
print('Errors for set with top 60 words:')
task32(x2training,x2validation,x2test,y_training)
print('Errors for set with top 160 words:')
task32(x3training,x3validation,x3test,y_training)

