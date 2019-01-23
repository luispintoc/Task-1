import json # we need to use the JSON package to load the data, since the data is stored in JSON format
import numpy as np
import matplotlib.pyplot as pt
from proj1_task1 import *
from proj1_task2 import closed_form

# It a list of data points, where each datapoint is a dictionary with the following attributes:
# popularity_score : a popularity score for this comment (based on the number of upvotes) (type: float)
# children : the number of replies to this comment (type: int)
# text : the text of this comment (type: string)
# controversiality : a score for how "controversial" this comment is (automatically computed by Reddit)
# is_root : if True, then this comment is a direct reply to a post; if False, this is a direct reply to another comment 

with open("proj1_data.json") as fp:
    data = json.load(fp)


#for Task 3.2
[x1,x2,x3,y_training] = proj1_task1.splitData(data,0,10000)
[x1_validation, x2_validation, x3_validation, y_validation] = proj1_task1.splitData(data,10001,11000)
[x1_test, x2_test, x3_test, y_test] = proj1_task1.splitData(data,11001,12000)

w1 = []
w2 = []
w3 = []
w1 = closed_form(x1, y_training)
w2 = closed_form(x2, y_training)
w3 = closed_form(x3, y_training)

y1_predicted = np.matmul(w1,x1_validation.T)
y_predicted2 = np.matmul(w,x_test.T)

y2_predicted = np.matmul(w2,x2_validation.T)
y_predicted2 = np.matmul(w,x_test.T)

y3_predicted = np.matmul(w3,x3_validation.T)
y_predicted2 = np.matmul(w,x_test.T)

#errors for the validation set
error1 = np.square(np.subtract(y1_predicted, y_validation)).mean()
print('The mean-squared error on the validation set with no text features is:', error1)

error2 = np.square(np.subtract(y2_predicted, y_validation)).mean()
print('The mean-squared error on the validation set including the top 60 words is:', error2)

error3 = np.square(np.subtract(y3_predicted, y_validation)).mean()
print('The mean-squared error on the validation set including the top 160 words is:', error3)

#errors for the test set
error4 = np.square(np.subtract(y1_predicted2, y_test)).mean()
print('The mean-squared error on the test set with no text features is:', error4)

error5 = np.square(np.subtract(y2_predicted2, y_test)).mean()
print('The mean-squared error on the test set including the top 60 words is:', error5)

error6 = np.square(np.subtract(y3_predicted2, y_test)).mean()
print('The mean-squared error on the test set including the top 160 words is:', error6)