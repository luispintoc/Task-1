import json
import numpy as np
import re as re 
import collections

cnt = collections.Counter()
text_list = [] 
sentence = [] 
is_root_list = []
popularity_list = [] 
controversiality_list = []
children_list = []  
comments_list = [] 

with open("proj1_data.json") as fp:
    data = json.load(fp)

def boolToBinary(feature): 
    if feature is False:
        return 0
    else: 
	    return 1

def dictToMatrix (popList, text_data): #first input: N top words , second input: comments
    
    X = np.zeros( (len(text_data), len(popList)) )
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
            column += 1
        row += 1
    return X

def newDict(text_list):
        for sentence in text_list:
            for word in sentence: 
                cnt[word] += 1
        #print (cnt)

def topNwords(N):
    finalList = [] 
    topNWordsList = cnt.most_common(N)
    print("topNWordsList ", topNWordsList)
    for (word, value) in topNWordsList: 

        finalList.append(word)
    print (finalList)
    return finalList

def splitData(data):
    i = 0 
    while i < 1000:
        is_root = boolToBinary(data[i]['is_root'])
        is_root_list.append(is_root)
        popularity_list.append(data[i]['popularity_score'])
        comments_list.append(data[i]['text'])
        controversiality_list.append(data[i]['controversiality'])
        children_list.append(data[i]['children'])
        
        text_list.append(data[i]['text'].lower().split())
        
        i += 1
    newDict(text_list)


#x = [] 
#print("text_list \n" , text_list)
#print("comments_list \n" , comments_list)
#top3Words = dictToMatrix(topNwords(3), text_list)
#print("top3Words \n" , top3Words)

def runIt(taskNumber):
    y = []
    y = np.zeros((len(popularity_list),1))
    row = 0

    for value in popularity_list:
        y[row,0] = value
        row += 1

    if taskNumber == 'Task3.2':
        #Use this x for Task 3.2
        splitData(data)
        
                
        top60_words_counts = dictToMatrix(topNwords(60), text_list)
        top160_words_counts = dictToMatrix(topNwords(160),text_list)

        #print(top60_words_counts[0:10])
        #print(topNwords(dict,60))
        #print(comments_list[0][1])
        
        x_no_text = np.column_stack((children_list,controversiality_list,is_root_list))
        x_top_60 = np.column_stack((children_list,controversiality_list,is_root_list,top60_words_counts))
        x_top_160 = np.column_stack((children_list,controversiality_list,is_root_list,top160_words_counts))
        return (x_no_text, x_top_60, x_top_160, y)