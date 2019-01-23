import numpy as np
import matplotlib.pyplot as pt

popList = ["they", "cool", "kids"]
text_data = [["they", "like", "chess"], ["cool", "kids", "play"], ["im", "cool", "cause", "they", "said"]]
dict = {}
text_list = []

X = np.zeros( (len(text_data), len(popList)) )

def filterOutPunc(text): 
    endCheck = len(text)-1
    tempWord = text
    if(text[endCheck] == '!' or text[endCheck] == '.' or text[endCheck] == '?'): 
        #print("found an !")
        tempWord = text[0:endCheck]
        #print("the word is:" + tempWord)
    if(text[0] == '"' ):
        #print("found a quote--front")
        tempWord = text[1:]
        #print(tempWord)
    if(text[endCheck] == '"' ):
        #print("found a quote--back")
        tempWord = text[:endCheck]
        #print(tempWord)
    if(text[0] == '"' and text[endCheck] == '"'):
        #print("single word in quotes")
        tempWord = text[1:endCheck]
    if(text[endCheck] == ',' ):
        #print("found a comma")
        tempWord = text[:endCheck]
    if(text[0] == '*' ):
        #print("found an asterisk--front")
        tempWord = text[1:]
    if(text[endCheck] == '*' ):
        #print("found an asterisk-back")
        tempWord = text[:endCheck]
    if(text[0] == '*' and text[endCheck] == '*'):
        #print("single word in asterisks")
        tempWord = text[1:endCheck]
    
    return tempWord       


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



def putInDict(text_list):
    for text in text_list: 
        newWord = filterOutPunc(text)
        if newWord in dict: 
            counter = dict[newWord] +1
            dict[newWord] = counter 
        else: 
            #newWord = filterOutPunc(text)
            dict[newWord] = 1

i = 0
while i < 10000:

	
	text_list.append(data[i]['text'].lower().split())
	sentence.append(text_list[0][0])
	putInDict(sentence)
	
	i += 1