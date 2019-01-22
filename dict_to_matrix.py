import numpy as np
import matplotlib.pyplot as pt

popList = ["they", "cool", "kids"]
text_data = [["they", "like", "chess"], ["cool", "kids", "play"], ["im", "cool", "cause", "they", "said"]]


X = np.zeros( (len(text_data), len(popList)) )

def dictToMatrix (X, popList, text_data): 
    row = 0
    column = 0
    for sentence in text_data: 
        column = 0
        for entry in popList: 
            counter = 0
            for word in sentence:
                if word == entry:
                    counter += 1
                
            X[row, column] = counter
            #print(X)
            column += 1
        row += 1
    print(X)
  


dictToMatrix(X, popList, text_data) 