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


def task32(x_tr, y_tr, x_v, y_v):
    # add bias
    one1 = np.ones(len(x_tr))
    x_tr = np.column_stack((x_tr, one1))
    one1 = np.ones(len(x_v))
    x_v = np.column_stack((x_v, one1))

    print('Number of columns: ',len(x_tr[0]))
    print('Rank of x_tr: ', np.linalg.matrix_rank(x_tr))

    # solve
    w = closed_form(x_tr, y_tr)
    y_predicted = np.matmul(w.T, x_v.T)
    #errors for the validation set
    error = np.square(np.subtract(y_predicted, y_v)).mean()
    print('The MSE on the validation set: ', error)

    
#for Task 3.2
[x1_tr, x2_tr, x3_tr, y_tr] = proj1_task1.splitData(data,0,10000,'Task3.2')
[x1_v, x2_v, x3_v, y_v] = proj1_task1.splitData(data,10001,11000,'Task3.2')
#[x1_test, x2_test, x3_test, y_test] = proj1_task1.splitData(data,11001,12000,'Task3.2')


#task32(x1_tr, y_tr, x1_v, y_v)
task32(x2_tr, y_tr, x2_v, y_v)
#task32(x3_tr, y_tr, x3_v, y_v)
'''
for j in range(4,64):
    c = 0
    for i in range(0, len(x1_tr)):
        c += x2_tr[i][j]
    print(c)
    
'''













        
