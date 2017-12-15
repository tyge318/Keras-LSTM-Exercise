import os, sys, string
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import nltk
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize
from keras.preprocessing.sequence import pad_sequences
import pickle

with open('valid_words.pickle', 'rb') as f:
    valid_words = pickle.load(f)
exclude = set(string.punctuation)
raw_texts = open('CPRFD10.TXT', encoding='utf-8').read()
raw_texts = raw_texts.strip().lower()
raw_texts = ''.join(ch for ch in raw_texts if ch not in exclude and type(ch) != 'str')
raw_words = nltk.word_tokenize(raw_texts)
raw_words = [w if w in valid_words else 'UNK' for w in raw_words]

words = sorted(list(set(raw_words)))
word_to_index = {w: i+1 for i, w in enumerate(words)}
index_to_word = {i+1: w for i, w in enumerate(words)}

n_vocab = len(words)+1
print('Total number of words:', n_vocab)

sent_len = 10
dataX, dataY = [], []
for i in range(len(raw_words) - sent_len):
    dataX.append([word_to_index[w] for w in raw_words[i:i+sent_len]])
    dataY.append(word_to_index[raw_words[i+sent_len]])
n_patterns = len(dataX)
print('Total Patterns:', n_patterns)

X = np.reshape(dataX, (n_patterns, sent_len, 1))
X = X/float(n_vocab)
y = np_utils.to_categorical(dataY)

# 3 layer stacked LSTM model
model = Sequential()
model.add(LSTM(1024, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(1024))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model = load_model('weights-improvement-50-0.2742.hdf5')
model.compile(loss='categorical_crossentropy', optimizer='adam')


# define the checkpoint
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# training (fit the model)
model.fit(X, y, epochs=50, batch_size=1024, callbacks=callbacks_list)
