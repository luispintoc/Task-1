import json
import numpy as np
import matplotlib.pyplot as pt
from proj1_task1 import *
from proj1_task2 import closed_form

with open("proj1_data.json") as fp:
    data = json.load(fp)

w4 = [] 

[x4, y_training] = proj1_task1.splitData(data,0,1000,'Task3.3')
[x4_validation, y_validation] = proj1_task1.splitData(data,1001,1200,'Task3.3')
[x4_test, y_test] = proj1_task1.splitData(data,1101,1300,'Task3.3')
y4_predicted = np.matmul(w4,x4_validation.T)


#errors 
error4 = np.square(np.subtract(y4_predicted, y_validation)).mean()
