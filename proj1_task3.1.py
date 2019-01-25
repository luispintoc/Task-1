import json # we need to use the JSON package to load the data, since the data is stored in JSON format
import numpy as np
import matplotlib.pyplot as pt
from proj1_task1 import splitData
from proj1_task2 import closed_form
from proj1_task2 import grad_des
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


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


#def task32 (change)


[x_training, y_training] = splitData(data,1001,11000,'Task3.1')

one1 = np.ones(len(x_training))
newx_training = np.column_stack((x_training,one1))

[x_validation, y_validation] = splitData(data,11001,12000,'Task3.1')

one2 = np.ones(len(x_validation))
newx_validation = np.column_stack((x_validation,one2))

[x_test, y_test] = splitData(data,0,1000,'Task3.1')

one3 = np.ones(len(x_test))
newx_test = np.column_stack((x_test,one3))


#Closed_form approach
w = []
w = closed_form(newx_training, y_training)


yclose_predicted_val = np.matmul(newx_validation,w)
yclose_predicted_test = np.matmul(newx_test,w)


close_error_val = np.square(np.subtract(yclose_predicted_val, y_validation)).mean()
print('The mean-squared error on the validation set is:', close_error_val)

close_error_test = np.square(np.subtract(yclose_predicted_test, y_test)).mean()
print('The mean-squared error on the test set is:', close_error_test)



# lm2 = linear_model.LinearRegression()

# lm2.fit(newx_training,y_training)
# pr = lm2.predict(newx_validation)
# errorsk = np.square(np.subtract(pr, y_validation)).mean()
#print('The mean-squared error using sklearn with no text features is:', errorsk)

'''

#Gradient descent approach
wd = []
w0 = np.random.random((len(newx_training[0]),1))
beta = list(range(0,len(newx_training)))
# print(len(w0))
# print(len(x_training))
wd = grad_des(newx_training,y_training,w0,beta,0.00001,0.000001,0) #X, Y, w0, beta, eta0, eps, 

ygrad_predicted_val = np.matmul(newx_validation,wd)

ygrad_predicted_test = np.matmul(newx_test,wd)

grad_error_validation = np.square(np.subtract(ygrad_predicted_val, y_validation)).mean()
print('The mean-squared error on the validation set is:', grad_error_validation)
grad_error_test = np.square(np.subtract(ygrad_predicted_test, y_test)).mean()
print('The mean-squared error on the test set is:', grad_error_test)

'''




