# Now the data is loaded.
# It a list of data points, where each datapoint is a dictionary with the following attributes:
# popularity_score : a popularity score for this comment (based on the number of upvotes) (type: float)
# children : the number of replies to this comment (type: int)
# text : the text of this comment (type: string)
# controversiality : a score for how "controversial" this comment is (automatically computed by Reddit)
# is_root : if True, then this comment is a direct reply to a post; if False, this is a direct reply to another comment 

import json # we need to use the JSON package to load the data, since the data is stored in JSON format
import numpy as np
import matplotlib.pyplot as pt

with open("proj1_data.json") as fp:
    data = json.load(fp)

## VARAIBLES ##

is_root_list = []
popularity_list = []
controversiality_list = []
children_list = []
words = []
sentence = []

## FUNCTIONS ##

def bool_to_binary(feature):
	if feature is False:
		return 0
	else: 
		return 1

def sort_dict(dict):
	sorted_d = sorted(dict.items(), key=lambda x:x[1])
	return sorted_d

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

dict = {}

def putInDict(text_list):
    for text in text_list: 
        newWord = filterOutPunc(text)
        if newWord in dict: 
            counter = dict[newWord] +1
            dict[newWord] = counter 
        else: 
            #newWord = filterOutPunc(text)
            dict[newWord] = 1

wordMatrix = np.eye(1,160) #1 row, 160 columns 

row = 1
column = 1
counter = 0
def makeMatrix(wordMatrix, dict, text_list):
    for word in dict: #iterate through dictionary
        for sentence in text_list: 
            for text in sentence:
                if word == text: 
                    wordMatrix[row, column] += 1

                 


## LOOPS ##

i = 0
while i < 1000:
	
	is_root = bool_to_binary(data[i]['is_root'])
	is_root_list.append(is_root)
	
	popularity_list.append(data[i]['popularity_score'])

	controversiality_list.append(data[i]['controversiality'])

	children_list.append(data[i]['children'])

	text_list = []
	text_list.append(data[i]['text'].lower().split())
	sentence.append(text_list[0][0])
	putInDict(sentence)
	
	i += 1


print(sort_dict(dict)) 




## PLOTS ##

pt.figure(figsize=(10,4))
pt.subplot(1,3,1)
pt.scatter(is_root_list,popularity_list,s=3,c='c')
#pt.title('Popularity vs is_root for training data')
pt.xlabel('is_root')
pt.ylabel('Popularity_score')
#pt.show()

pt.subplot(1,3,2)
pt.scatter(controversiality_list,popularity_list,s=3,c='r')
#pt.title('Popularity vs Controversiality for training data')
pt.xlabel('Controversiality')
pt.ylabel('Popularity_score')
#pt.show()

pt.subplot(1,3,3)
pt.scatter(children_list,popularity_list,s=3,c='b')
#pt.title('Popularity vs children for training data')
pt.xlabel('children')
pt.ylabel('Popularity_score')
#pt.show()

#print(sentence)