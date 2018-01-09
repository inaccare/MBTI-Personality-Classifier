#!/usr/bin/env python

import numpy
import random
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing import text

TEST_BATCH_SIZE = 100
NUM_EPOCHS = 10
TOP_WORDS = 2000 # the number of the most common words that will be given a unique identifier for the word vector
LETTER_PAIRS = [['I', 'E'], ['N', 'S'], ['F', 'T'], ['P', 'J']]
DIMENSIONS = ['IE', 'NS', 'FT', 'PJ']

# load the dataset
for k in range(len(DIMENSIONS)):
	x_train = [] 
	y_train = [] 
	x_test = [] 
	y_test = []

	with open('train_' + LETTER_PAIRS[k][0] + '.csv', 'r') as f: 
		for row in f:
			x_train.append(row)
			y_train.append(0)

	with open('train_' + LETTER_PAIRS[k][1] + '.csv', 'r') as f: 
		for row in f:
			x_train.append(row)
			y_train.append(1)

	with open('test_' + LETTER_PAIRS[k][0] + '.csv', 'r') as f: 
		for row in f: 
			x_test.append(row)
			y_test.append(0)

	with open('test_' + LETTER_PAIRS[k][1] + '.csv', 'r') as f: 
		for row in f: 
			x_test.append(row)
			y_test.append(1)

	# Shuffle x_train and y_train in the same (random) order 
	c = list(zip(x_train, y_train))
	random.shuffle(c)
	x_train, y_train = zip(*c)
	x_train = list(x_train)
	y_train = list(y_train)

	# tokenize
	tokenizer = text.Tokenizer(num_words=TOP_WORDS, lower=True, split=' ')
	tokenizer.fit_on_texts(x_train)
	x_train = tokenizer.texts_to_sequences(x_train)
	x_test = tokenizer.texts_to_sequences(x_test)

	# truncate and pad input sequences
	max_post_length = 80
	x_train = sequence.pad_sequences(x_train, maxlen=max_post_length)
	x_test = sequence.pad_sequences(x_test, maxlen=max_post_length)

	# create the model
	embedding_vector_length = 20
	model = Sequential()
	model.add(Embedding(TOP_WORDS, embedding_vector_length, input_length=max_post_length))
	model.add(LSTM(embedding_vector_length, dropout=0.2, recurrent_dropout=0.2))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	model.fit(x_train, y_train, nb_epoch=NUM_EPOCHS, batch_size=32)

	# final evaluation of the model
	NUM_CORRECT = 0
	NUM_WRONG = 0
	for i in range(int(len(x_test)/TEST_BATCH_SIZE)):
		a = i*TEST_BATCH_SIZE
		b = a+TEST_BATCH_SIZE
		acc = accuracy_score(y_test[a:b], model.predict_classes(x_test[a:b]))
		if acc >= 0.5: 
			NUM_CORRECT += 1
		else: 
			NUM_WRONG += 1

	# print raw test accuracy
	raw_score = accuracy_score(y_test, model.predict_classes(x_test))
	print(model.predict_classes(x_test))
	print("Test Accuracy on Individual Posts: %.2f%%" % (raw_score*100))

	# save test accuracy to .txt file
	results_path = 'accuracy_results_' + DIMENSIONS[k] + '.txt'
	with open(results_path, 'w') as f:
		f.write('# of Batches Correct: ' + str(NUM_CORRECT) + '\n')
		f.write('# of Batchs Wrong: ' + str(NUM_WRONG) + '\n')
		f.write('Proportion of Batches Correct: ' + str(float(NUM_CORRECT/(NUM_CORRECT+NUM_WRONG))) + '\n')
		f.write('Proportion of Individual Posts Correct: ' + str(raw_score) + '\n')

	# save trained model to .h5 file
	path = 'model_' + DIMENSIONS[k]+ '.h5'
	model.save(path)