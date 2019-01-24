#All functions are within splitData
#Inputs are: data set, first data point and last data point (to differenciate between training, validat and test sets)
#Outputs: There are two different outputs depending on Task 3
#For Task3.1 we ommit the text features, therefore the outputs are X(children,controversiality,is_root) and Y
#For Task3.2 we use the text features and we have 3 different X sets( one without those features,
#one with the top 60 and one with the top 160)


import numpy as np
import matplotlib.pyplot as pt

class proj1_task1:

	def splitData(data,first_datapoint,last_datapoint,TaskNumber): 
	#for training (0,10000) - for validation (10001,11000) and testing (11001,12000)
		i = first_datapoint
		
		## VARAIBLES ##
		is_root_list = []
		popularity_list = []
		controversiality_list = []
		children_list = []
		comments_list = []
		words = []
		sentence = []

		def bool_to_binary(feature):
			if feature is False:
				return 0
			else: 
				return 1

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


		while i < last_datapoint:
			
			is_root = bool_to_binary(data[i]['is_root'])
			is_root_list.append(is_root)
			
			popularity_list.append(data[i]['popularity_score'])

			comments_list.append(data[i]['text'])

			controversiality_list.append(data[i]['controversiality'])

			children_list.append(data[i]['children'])

			text_list = []
			text_list.append(data[i]['text'].lower().split())
			sentence.append(text_list[0][0])
			putInDict(sentence)
			
			i += 1


		def topNwords(dict,N):
		    sorted_d = sorted(dict.items(), key=lambda x:x[1])
		    i = 0
		    topwords = []
		    while i < N:
		        topwords.append(sorted_d[len(sorted_d)-1-i])
		        i += 1
		    return topwords


		def dictToMatrix (popList, text_data): #first input: N top words , second input: comments
		    X = []
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


		#To see the top words
		#r = topNwords(dict,160)
		#return (x,y,r)

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

		
		elif TaskNumber == 'Task3.2':
			#Use this x for Task 3.2
			
			top60_words = dictToMatrix(topNwords(dict,60),comments_list)
			top160_words = dictToMatrix(topNwords(dict,160),comments_list)

			x_no_text = np.column_stack((children_list,controversiality_list,is_root_list))
			x_top_60 = np.column_stack((children_list,controversiality_list,is_root_list,top60_words))
			x_top_160 = np.column_stack((children_list,controversiality_list,is_root_list,top160_words))
			return (x_no_text, x_top_60, x_top_160, y)
		
		else:
			#Use this for x for Task 3.3
			x = np.column_stack((children_list,controversiality_list,is_root_list)) #not correct
			return(x, y)



