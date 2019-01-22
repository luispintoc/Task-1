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

def bool_to_binary(feature):
	if feature is False:
		return 0
	else: 
		return 1

i = 0
is_root_list = []
popularity_list = []
controversiality_list = []
children_list = []
text_list = []
words = []



while i < 10000:
	
	is_root = bool_to_binary(data[i]['is_root'])
	is_root_list.append(is_root)
	
	popularity_list.append(data[i]['popularity_score'])

	controversiality_list.append(data[i]['controversiality'])

	children_list.append(data[i]['children'])

	text_list.append(data[i]['text'].lower().split())
	
	i += 1

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
pt.show()

#print(text_list)