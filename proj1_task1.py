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

def splitData(data,first_datapoint,last_datapoint,taskNumber):

	## VARAIBLES ##
	cnt = collections.Counter()
	cnt_feat = collections.Counter()
	is_root_list = []
	popularity_list = []
	controversiality_list = []
	children_list = []
	comments_list = [] 
	words = []
	#sentence = []
	extLinkCount = [] 
	text_list = []

	def bool_to_binary(feature):
			if feature is False:
					return 0
			else: 
					return 1

	def newDict(text_list):
		for sentence in text_list:
			for word in sentence: 
				cnt[word] += 1


	def filterOutPunc(text_list):
		newTextList = [] 
		for sentence in text_list:
			newSentence = []   
			for text in sentence: 
				endCheck = len(text)-1
				tempWord = text
				if(text[endCheck] == '!' or text[endCheck] == '.' or text[endCheck] == '?'): 
					print("1")
					tempWord = text[0:endCheck]     
				if(text[0] == '"' ):
					print("2")
					tempWord = text[1:]
					
				if(text[endCheck] == '"' ):
					print("3")
					tempWord = text[:endCheck]
					
				if(text[0] == '"' and text[endCheck] == '"'):
					print("4")
					tempWord = text[1:endCheck]
				if(text[endCheck] == ',' ):
					
					tempWord = text[:endCheck]
				if(text[0] == '*' ):
					
					tempWord = text[1:]
				if(text[endCheck] == '*' ):
					
					tempWord = text[:endCheck]
				if(text[0] == '*' and text[endCheck] == '*'):
					
					tempWord = text[1:endCheck]

				newSentence.append(tempWord)
			newTextList.append(newSentence)
		return newTextList       

	def hasExternalLink(text_list):
		for sentence in text_list: 
			counter = 0
			for text in sentence:
				if (text[0:3] == "www"):
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
		
		#if(taskNumber == 'Task 3.3'): 
			#filterOutPunc(text_list)
		
		i += 1
	newDict(text_list)
	extLinkCount.append(hasExternalLink(text_list))


	def topNwords(N):
		finalList = []
		topNWordsList = cnt.most_common(N)
		#print(topNWordsList)
		for (word,value) in topNWordsList:
			finalList.append(word)
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
		print(text_list)
		return (x, y)		

	
	if taskNumber == 'Task3.2':
		#Use this x for Task 3.2
		x_no_text = np.column_stack((children_list,controversiality_list, is_root_list))

		top60_words = dictToMatrix(topNwords(60),text_list)
		top160_words = dictToMatrix(topNwords(160),text_list)
				
		x_top_60 = np.column_stack((x_no_text,top60_words))
		x_top_160 =  np.column_stack((x_no_text,top160_words))
		return (x_no_text, x_top_60, x_top_160, y)

	if taskNumber == 'Task3.3':
	# 	#Use this for x for Task 3.3
	
		x_no_text = np.column_stack((children_list,controversiality_list, is_root_list))
		#externalList feature
		x_with_externList = np.column_stack((x_no_text, extLinkCount))
		#more accurate top words reading 
		filterOutPunc(text_list)
		newDict(text_list)

		#top60_words = dictToMatrix(topNwords(60),text_list)
		top160_words = dictToMatrix(topNwords(160),text_list)
				
		#x_top_60 = np.column_stack((x_no_text,top60_words))
		x_top_160 =  np.column_stack((x_no_text,top160_words))

		x_all = np.column_stack((x_with_externList, x_top_160))

		return(x_no_text, x_all, y)
		

	 	#top60_words = dictToMatrix(topNwords(60),text_list)
	 	#top160_words = dictToMatrix(topNwords(160),text_list)
		
		#x_top_60 = np.column_stack((x_no_text,top60_words))
	 	#x_top_160 =  np.column_stack((x_no_text,top160_words))
	 	#return (x_no_text, x_top_60, x_top_160, y)



