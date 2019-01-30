'''
Mini-Project1 - COMP 551 - Winter 2019
Mahyar Bayran
Luis Pinto
Rebecca Salganik
'''

#All functions are within splitData
#Inputs are: data set, first data point and last data point (to differenciate between training, validat and test sets)
#Outputs: There are two different outputs depending on Task 3
#For Task3.1 we ommit the text features, therefore the outputs are X(children,controversiality,is_root) and Y
#For Task3.2 we use the text features and we have 3 different X sets( one without those features,
#one with the top 60 and one with the top 160)

import json
import numpy as np
import collections
import matplotlib.pyplot as pt
from nltk.util import ngrams
import operator
import collections

def splitData(data,first_datapoint,last_datapoint,taskNumber):

    ## VARAIBLES ##
    cnt = collections.Counter()
    cnt2 = collections.Counter()
    is_root_list = []
    popularity_list = []
    controversiality_list = []
    children_list = []
    comments_list = [] 
    words = []
    extLinks = [] 
    text_list = []
    extLinkCount = []
    sWords = ["once"]
    #, "about", "but", "again", "then", "that", "a", "i", "after", "it", "how", "if", "the", "in", 'the', 'i', 'to', 'and', 'a', 'of', 'it', 'you', 'that', 'in']
     
    def bool_to_binary(feature):
        if feature is False:
            return 0
        else: 
            return 1

    def newDict(text_list,cnt1):
        for sentence in text_list:
            for word in sentence: 
                #cnt[word] += 1
                cnt1[word] += 1


    def filterOutPunc(text): 
        endCheck = len(text)-1
        tempWord = text
        if(text[endCheck] == '!' or text[endCheck] == '.' or text[endCheck] == '?'): 
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

    def hasExternalLink(text_list):
        #extLinkCount = 0
        
        for sentence in text_list:
        	counter = 0
	        for text in sentence: 
	            if (text[0:3] == "htt"):
	                counter += 1
	        extLinkCount.append(counter)
        return(extLinkCount)


    #for training (0,10000) - for validation (10000,11000) and testing (11000,12000)
    i = first_datapoint		
    while i < last_datapoint:
                    
        is_root = bool_to_binary(data[i]['is_root'])
        is_root_list.append(is_root)
            
        popularity_list.append(data[i]['popularity_score'])

        controversiality_list.append(data[i]['controversiality'])

        children_list.append(data[i]['children'])

        text_list.append(data[i]['text'].lower().split())
            #sentence.append(text_list[0][0])
            #extLinks.append(text_list)
            
        i += 1
    newDict(text_list,cnt)
    links_list = hasExternalLink(text_list)


    def topNwords(N,cnt1):
        finalList = []
        topNWordsList = cnt1.most_common(N)
        for (word,value) in topNWordsList:
            finalList.append(word)
        return finalList

    def filterStopWords(text_list,sWords,cnt2):
    	for sentence in text_list:
    		for words in sentence:
    			if not words in sWords:
    				cnt2[words] += 1

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


    def extractBigrams (comments,n):
        cm_bi = []
        all_bi = {}
        top_bi = []
        
        for cm in comments:            
            cm_bi.append(list(ngrams(cm, 2)))
            for bi in list(ngrams(cm, 2)):
                if bi not in list(all_bi.keys()):
                    all_bi[bi] = 0
                else:
                    all_bi[bi] = all_bi[bi] + 1
        # ascending order
        sorted_bi = sorted(all_bi.items(), key=operator.itemgetter(1))
        # descending order
        sorted_bi.reverse()
        for i in range(0,n):
            top_bi.append(sorted_bi[i][0])        
        return dictToMatrix(top_bi, cm_bi)
    #loop to create y matrix 
    y = []
    y = np.zeros((len(popularity_list),1))
    row = 0
    for word in popularity_list:
        y[row,0] = word
        row += 1

    if taskNumber == 'Task3.1':		
        #Use this X for Task 3.1
        x = np.column_stack((children_list,controversiality_list,is_root_list))
        return (x, y)		
    
    if taskNumber == 'Task3.2':
        #Use this x for Task 3.2
        x_no_text = np.column_stack((children_list,controversiality_list, is_root_list))
        top60_words = dictToMatrix(topNwords(60,cnt),text_list)
        top160_words = dictToMatrix(topNwords(160,cnt),text_list)
        x_top_60 = np.column_stack((x_no_text,top60_words))
        x_top_160 =  np.column_stack((x_no_text,top160_words))
        return (x_no_text, x_top_60, x_top_160, y)
        
    if taskNumber == 'Task3.3':
        #Use this for x for Task 3.3
        x_no_text = np.column_stack((children_list,controversiality_list, is_root_list))
        #filterStopWords(text_list, sWords, cnt2)
        top60_words = dictToMatrix(topNwords(60,cnt),text_list)
        top_bigrams_counts = []
        top_bigrams_counts =  extractBigrams(text_list, 30)
        return (x_no_text, top_bigrams_counts , top60_words , y)
