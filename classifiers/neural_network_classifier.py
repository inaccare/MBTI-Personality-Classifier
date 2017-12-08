#!/usr/bin/env python

import random
import collections
import csv
import nltk
import numpy as np
from sklearn.neural_network import MLPClassifier

###############################
### CONSTANTS AND VARIABLES ###
###############################

TEST_BATCH_SIZE = 100
WORD_2_VEC_TRAIN_TEXT = "glove.6B.100d.txt"
LETTER_PAIRS = [['I', 'E'], ['N', 'S'], ['F', 'T'], ['P', 'J']]
DIMENSIONS = ['IE', 'NS', 'FT', 'PJ']
CORRECT = 0
WRONG = 0

w2v = None
with open(WORD_2_VEC_TRAIN_TEXT, "rb") as lines:
    w2v = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}

clf = MLPClassifier(solver='lbfgs', alpha=1e-2, hidden_layer_sizes=(5, 2), random_state=1)

dim = len(w2v.itervalues().next())

#################
### FUNCTIONS ###
#################

def mean_vector(words):
	result = np.zeros(dim)
	n = len(words)
	for w in words:
		if w in w2v:
			add = w2v[w]
		else: 
			add = np.zeros(dim)
		for i in range(0, dim):
			result[i] += add[i]
	for i in range(0, dim):
		result[i] = float(result[i]/n)
	return result

def makeTrain(filename, y):
	for_train = []
	with open(filename, 'r') as f:
		for row in f:
			sentences = []
			words = []
			tokenized = []
			try:
				sentences = nltk.sent_tokenize(row)
				for sent in nltk.sent_tokenize(row):
					words += nltk.word_tokenize(sent)
				check = True
			except UnicodeDecodeError: ### Some posts are untokenizable because of this, so we apply a special token to represent them 
				check = False
			if check:		
				for word in words:
					if ('http' not in word) and ('.com/' not in word):
						tokenized.append(word)
					else: 
						tokenized.append('WEBSITE_TOKEN') ### Replace websites with a website token 
				for_train.append((tokenized, y))
			else: 
				for_train.append(('POST_CONTAINS_ASCII_ERROR', y)) ### This is the special token
	return for_train

def makeTest(filename):
	for_test = []
	with open(filename, 'r') as f:
		i = 0
		for_append = []
		for row in f:
			i += 1
			sentences = []
			words = []
			tokenized = []
			try:
				sentences = nltk.sent_tokenize(row)
				for sent in nltk.sent_tokenize(row):
					words += nltk.word_tokenize(sent)
				check = True
			except UnicodeDecodeError: ### Some posts are untokenizable because of this, so we apply a special token to represent them 
				check = False
			if check:		
				for word in words:
					if ('http' not in word) and ('.com/' not in word):
						tokenized.append(word)
					else: 
						tokenized.append('WEBSITE_TOKEN') ### Replace websites with a website token
				for_append.append(tokenized)
			else: 
				for_append.append('POST_CONTAINS_ASCII_ERROR') ### This is the special token
			if i%TEST_BATCH_SIZE == 0: 
				for_test.append(for_append)
				for_append = []
	return for_test

def training(train):
	x_list = []
	y_list = []
	for x, y in train:
		x_list.append(x)
		y_list.append(y)
	clf.fit(x_list, y_list)

def testing(x, y):
	y_predictions = clf.predict(x)
	print y_predictions
	balance = 0
	for i in range(0, len(y_predictions)):
		balance += y_predictions[i]
	if balance >= 0: return 1
	else: return -1

def process(a, b, ab):	
	PATH_a_train = 'train_' + a + '.csv'
	PATH_b_train = 'train_' + b + '.csv'
	PATH_a_test = 'test_' + a + '.csv'
	PATH_b_test = 'test_' + b + '.csv'
	PATH_error = 'error_analysis_' + ab + '.txt'

	### TRAINING:
	train_a = makeTrain(PATH_a_train, -1)
	train_b = makeTrain(PATH_b_train, 1)
	train_ab = train_a + train_b
	train_ab_vec = []

	for train in train_ab:
		train_ab_vec.append((mean_vector(train[0]), train[1]))
	
	random.shuffle(train_ab_vec)

	training(train_ab_vec)

	### TESTING:
	test_a = makeTest(PATH_a_test)
	test_b = makeTest(PATH_b_test)
	test_ab = []
	
	test_vec_a = []
	test_vec_b = []

	for batch in test_a:
		vectorized_batch = []
		for post in batch:
			vectorized_batch.append(mean_vector(post))
		test_vec_a.append(vectorized_batch)

	for batch in test_b:
		vectorized_batch = []
		for post in batch:
			vectorized_batch.append(mean_vector(post))
		test_vec_b.append(vectorized_batch)

	correct_a = 0
	correct_b = 0
	wrong_a = 0
	wrong_b = 0

	out = open(PATH_error, 'w')

	for vec in test_vec_a:
		if (testing(vec, -1) == -1): 
			correct_a += 1
		else: 
			wrong_a += 1

	for vec in test_vec_b:
		if (testing(vec, 1) == 1): 
			correct_b += 1
		else: 
			wrong_b += 1

	print >> out, 'Total: ' + str(correct_a+correct_b) + ' / ' + str(correct_a+correct_b+wrong_a+wrong_b)
	print >> out, str(a) + ': ' + str(correct_a) + ' / ' + str(correct_a+wrong_a)
	print >> out, str(b) + ': ' + str(correct_b) + ' / ' + str(correct_b+wrong_b)
	out.close()

###########
### RUN ###
###########

for index in range(0, len(DIMENSIONS)):
	a = LETTER_PAIRS[index][0]
	b = LETTER_PAIRS[index][1]
	ab = DIMENSIONS[index] 
	process(a, b, ab)
