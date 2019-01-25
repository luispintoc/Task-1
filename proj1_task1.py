#All functions are within splitData
#Inputs are: data set, first data point and last data point (to differenciate between training, validat and test sets)
#Outputs: There are two different outputs depending on Task 3
#For Task3.1 we ommit the text features, therefore the outputs are X(children,controversiality,is_root) and Y
#For Task3.2 we use the text features and we have 3 different X sets( one without those features,
#one with the top 60 and one with the top 160)


import json
import numpy as np
import re as re 
import collections
import matplotlib.pyplot as pt

with open("proj1_data.json") as fp:
    data = json.load(fp)

def splitData(data,first_datapoint,last_datapoint,TaskNumber):
			
	## VARAIBLES ##
	cnt = collections.Counter()
	is_root_list = []
	popularity_list = []
	controversiality_list = []
	children_list = []
	comments_list = [] 
	words = []
	sentence = []
	extLinks = [] 

	def bool_to_binary(feature):
			if feature is False:
					return 0
			else: 
					return 1

	def newDict(text_list):
		for sentence in text_list:
			for word in sentence: 
				cnt[word] += 1


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
			extLinkCount = 0
			
			for text in text_list: 
					if (text[0:2] == "www"):
							extLinkCount += 1
					
			return(extLinkCount)

	#for training (0,10000) - for validation (10001,11000) and testing (11001,12000)
	i = first_datapoint		
	while i < last_datapoint:
			
		is_root = bool_to_binary(data[i]['is_root'])
		is_root_list.append(is_root)
		
		popularity_list.append(data[i]['popularity_score'])

		#cm = data[i]['text']
		#cm.encode('UTF-8', 'ignore').decode('UTF-8')
		#comments_list.append(filterOutPunc(cm.split(' ')))

		controversiality_list.append(data[i]['controversiality'])

		children_list.append(data[i]['children'])

		text_list = []
		text_list.append(data[i]['text'].lower().split())
		sentence.append(text_list[0][0])
		#extLinks.append(hasExternalLink(sentence))
		
		i += 1
	newDict(text_list)


	def topNwords(N):
   		finalList = [] 
    	topNWordsList = cnt.most_common(N)
    	print("topNWordsList ", topNWordsList)
    	for (word, value) in topNWordsList: 

        	finalList.append(word)
    	print (finalList)
    	return finalList
			
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


	
	#isolating y matrix from data 
	y = []
	y = np.zeros((len(popularity_list),1))
	row = 0

	for word in popularity_list:
			y[row,0] = word
			row += 1

	if TaskNumber == 'Task3.1':		
	
			#Use this X for Task 3.1
			x = np.column_stack((children_list,controversiality_list,is_root_list))
			return (x, y)		

	
	if TaskNumber == 'Task3.2':
			#Use this x for Task 3.2
			
					
			top60_words_counts = dictToMatrix(topNwords(60), text_list)
        	top160_words_counts = dictToMatrix(topNwords(160),text_list)

			#print(top60_words_counts[0:10])
			#print(topNwords(dict,60))
			#print(comments_list[0][1])
			
			x_no_text = np.column_stack((children_list,controversiality_list,is_root_list))
			x_top_60 = np.column_stack((children_list,controversiality_list,is_root_list,top60_words_counts))
			x_top_160 = np.column_stack((children_list,controversiality_list,is_root_list,top160_words_counts))
			return (x_no_text, x_top_60, x_top_160, y)
	
	if TaskNumber == 'Task3.3':
			#Use this for x for Task 3.3
	
			x = np.column_stack((children_list,controversiality_list,is_root_list,extLinks))
			return(x, y)



