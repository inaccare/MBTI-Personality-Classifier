#!/usr/bin/env python

import random
import collections
import numpy as np
import pandas as pd
import csv

letters = ('I', 'E', 'N', 'S', 'T', 'F', 'P', 'J')
df = pd.read_csv('mbti_1.csv')
train = collections.defaultdict(list) 
test = collections.defaultdict(list) 
set_types = {'train': train, 'test': test}

### make 80% of the data into training set, other 20% into test set
for row in df.iterrows():
	num = random.randint(1, 101)
	if num <= 80: 
		posts = row[1]['posts'].split('|||')
		train[row[1]['type']].append(posts)
	else:
		posts = row[1]['posts'].split('|||')
		test[row[1]['type']].append(posts)

### write csv files for every every letter class and train vs. test class (16 total)
for set_type in set_types.keys():
	for letter in letters: 
		PATH = set_type + '_' + letter + '.csv'
		with open(PATH, 'wb') as f:
			writer = csv.writer(f)
			for key in set_types[set_type].keys():
				if letter in key:
					for hundred_posts in set_types[set_type][key]:
						for post in hundred_posts:
							writer.writerow([post])



			
