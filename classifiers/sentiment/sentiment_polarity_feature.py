#!/usr/bin/env python

import collections
import feature_vector_classifier
from textblob import TextBlob

def extractWordFeatures(x):
    return {'asdf': x}

def makeTrain(for_train, filename):
    with open(filename, 'r') as f:
        for row in f:
            try:
                sentiment = TextBlob(row)
                for_train.append(sentiment.sentiment[0])
            except UnicodeDecodeError: 
                pass

def makeTest(for_test, filename):
    with open(filename, 'r') as f:
        i = 0
        append_row = []
        for row in f:
            try:
                i += 1
                sentiment = TextBlob(row)
                append_row.append(sentiment.sentiment[0])
            except UnicodeDecodeError: 
                pass
            if i%10 == 0: 
                for_test.append(append_row)
                append_row = []
            
        f.close()
            

feature_vector_classifier.run(extractWordFeatures, makeTrain, makeTest)