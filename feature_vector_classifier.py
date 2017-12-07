#!/usr/bin/env python

import random
import collections
import csv
import nltk
import enchant

###############################
### CONSTANTS AND VARIABLES ###
###############################

NUM_ITERS = 10
ETA = 0.01
TEST_BATCH_SIZE = 10

DIMENSIONS = ['IE', 'NS', 'FT', 'PJ']
LETTER_PAIRS = [['I', 'E'], ['N', 'S'], ['F', 'T'], ['P', 'J']]

BAG_OF_WORDS = True
WORD_LENGTH = False
NGRAMS = False
PART_OF_SPEECH_NGRAMS = False
PUNCTUATION = False
QUANTITATIVE = False
SPELLING = False

spellchecker = enchant.Dict("en_US")

#################
### FUNCTIONS ###
#################

def featureExtractor(x):
    features = collections.defaultdict(int)

    words = x.split()
    numWords = 0
    numPunctuation = 0
    numNumbers = 0
    numCorrectSpells = 0
    numMisspells = 0
    numDerogatory = 0

    for word in words:
        if ('http' not in word) and ('.com/' not in word):
            numWords += 1
            strippedWord = ''.join(char for char in word if char.isalpha())
            strippedWord = strippedWord.lower()

            # Bag of words
            if BAG_OF_WORDS:
            	features[strippedWord] += 1

            # Word length
            if WORD_LENGTH:
            	features[len(strippedWord)] += 1
            
            # Proportion of punctuation marks to words
            if (',' or '.' or '!' or '?' or ';' or ':' or '-' or '(' or ')' or '/' or '*' or '\'' or '\"') in word:
                numPunctuation += 1
                
            # Proportion of numbers to words
            if ('1' or '2' or '3' or '4' or '5' or '6' or '7' or '8' or '9') in word:
                numNumbers += 1
            
            # Proportion of misspells to correct spells
        	if (strippedWord != '') and (spellchecker.check(strippedWord)):
        		numCorrectSpells += 1
        	else:
        		numMisspells += 1
    
    if PART_OF_SPEECH_NGRAMS:
	    pos_n = 3
	    tags = []
	    try: 
	        for sent in nltk.sent_tokenize(x):
	            tags += nltk.pos_tag(nltk.word_tokenize(sent))
	    except UnicodeDecodeError:
	        pass
	    post_as_tags = []
	    for word, tag in tags:
	        post_as_tags.append(tag)
	    for i in xrange(pos_n, len(post_as_tags)):
	        current = ''
	        for j in range(i-pos_n,i+1):
	            try: 
	                current += post_as_tags[j]
	            except UnicodeDecodeError:
	                pass
	        features['pos_ngrams', current] += 1
    
    if NGRAMS:
	    n = 3
	    strippedX = ''.join(char for char in x if char.isalpha())
	    for i in xrange(n, len(strippedX)):
	        currentNGram = strippedX[i-n:i+1]
	        features['n-grams', currentNGram] += 1

    if numWords != 0:
        if PUNCTUATION: features[('special', 'punctiation to words')] = float(numPunctuation) / float(numWords)
        if QUANTITATIVE: features[('special', 'numbers to words')] = float(numNumbers) / float(numWords)
        if SPELLING: features[('special', 'misspells to spells')] = float(numMisspells) / float(numCorrectSpells)

    return features

def makeTrain(for_train, filename):
    with open(filename, 'r') as f:
        for row in f:
            for_train.append(row)

def makeTest(for_test, filename):
    with open(filename, 'r') as f:
        i = 1
        append_row = []
        for row in f:
            append_row.append(row)
            if i%TEST_BATCH_SIZE == 0: 
                for_test.append(append_row)
                append_row = []
            i += 1

def dotProduct(d1, d2):
	if len(d1) < len(d2):
		return dotProduct(d2, d1)
	else:
		return sum(d1.get(f, 0) * v for f, v in d2.items())

def increment(d1, scale, d2):
	for f, v in d2.items():
		d1[f] = d1.get(f, 0) + v * scale

def verbosePredict(phi, y, weights, out):
	yy = 1 if dotProduct(phi, weights) >= 0 else -1
	if y == yy: return 1
	else: return -1
	'''
	if y:
		print >>out, 'Truth: %s, Prediction: %s [%s]' % (y, yy, 'CORRECT' if y == yy else 'WRONG')
	else:
		print >>out, 'Prediction:', yy
	for f, v in sorted(phi.items(), key=lambda (f, v) : -v * weights.get(f, 0)):
		w = weights.get(f, 0)
		print >>out, "%-30s%s * %s = %s" % (f, v, w, v * w)
	'''

def addToTrain(train, for_train, y):
	for el in for_train:
		train.append((el, y))

def addToTest(test, for_test, y):
	for el in for_test:
		test.append((el, y))

def training(train, weights):
	for i in range(0, NUM_ITERS):
		count = 1
		for x, y in train:
			phi = featureExtractor(x)
			margin = dotProduct(weights, phi) * y
			if margin < 1:
				gradient = phi
				for key in gradient:
					gradient[key] = gradient[key]*-1*y
			else:
				gradient = {}
				for key in phi:
					gradient[key] = 0
			increment(weights, -ETA, gradient)
			count += 1

def testing(pairs, weights, PATH):
	out = open(PATH, 'w')
	for pair in pairs:
		user_posts, y = pair
		result_y = 0
		for post in user_posts:
			print >>out, '===', post
			result_y += verbosePredict(featureExtractor(post), y, weights, out)
		if result_y >= 0: prediction = y
		else: prediction = -1*y
		print >>out, 'Truth: %s, Prediction: %s [%s]' % (y, prediction, 'CORRECT' if prediction == y else 'WRONG')
	out.close()

def process(a, b, ab):
	train_a = []
	train_b = []

	test_a = []
	test_b = []

	train_ab = []
	test_ab = []

	weights_ab = {}

	PATH_a_train = 'train_' + a + '.csv'
	PATH_b_train = 'train_' + b + '.csv'

	PATH_a_test = 'test_' + a + '.csv'
	PATH_b_test = 'test_' + b + '.csv'

	PATH_error = 'error_analysis_' + ab + '.txt'

	makeTrain(train_a, PATH_a_train)
	makeTrain(train_b, PATH_b_train)

	makeTest(test_a, PATH_a_test)
	makeTest(test_b, PATH_b_test)

	addToTrain(train_ab, train_a, -1)
	addToTrain(train_ab, train_b, 1)

	addToTest(test_ab, test_a, -1)
	addToTest(test_ab, test_b, 1)

	random.shuffle(train_ab)
	random.shuffle(test_ab)

	training(train_ab, weights_ab) ### TRAINING POINT
	testing(test_ab, weights_ab, PATH_error) ### TESTING POINT

###########
### RUN ###
###########

for index in range(len(DIMENSIONS)):
	a = LETTER_PAIRS[index][0]
	b = LETTER_PAIRS[index][1]
	ab = DIMENSIONS[index] 
	process(a, b, ab)

