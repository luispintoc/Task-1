import json # we need to use the JSON package to load the data, since the data is stored in JSON format
import numpy as np
import matplotlib.pyplot as pt
from proj1_task1 import *
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
newx_training = []
newx_validation = []
newx_test = []

[x_training, y_training] = proj1_task1.splitData(data,0,10000)

one1 = np.ones(len(x_training))
newx_training = np.column_stack((x_training,one1))

[x_validation, y_validation] = proj1_task1.splitData(data,10001,11000)

one2 = np.ones(len(x_validation))
newx_validation = np.column_stack((x_validation,one2))

[x_test, y_test] = proj1_task1.splitData(data,11001,12000)

one3 = np.ones(len(x_test))
newx_test = np.column_stack((x_test,one3))


#Closed_form approach
w = []
w = closed_form(x_training, y_training)
y_predicted = np.matmul(w,x_validation.T)
y_predicted2 = np.matmul(w,x_test.T)

error = np.square(np.subtract(y_predicted, y_validation)).mean()
print('The mean-squared error on the validation set is:', error)

error2 = np.square(np.subtract(y_predicted2, y_test)).mean()
print('The mean-squared error on the test set is:', error2)

#Gradient descent approach
#wd = []
#wd = grad_des(x_training,)
