import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import nltk
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize
from keras.preprocessing.sequence import pad_sequences

raw_texts = open("wonderland.txt").read()
raw_texts = raw_texts.strip().lower()

sentences = list(map(lambda x: nltk.word_tokenize(x), sent_tokenize(raw_texts)))
sentences = [['@']+sent+['#'] for sent in sentences]

words = [word for sentence in sentences for word in sentence]
words = sorted(list(set(words)))
word_to_index = {w: i+1 for i, w in enumerate(words)}
index_to_word = {i+1: w for i, w in enumerate(words)}

n_vocab = len(words)+1
print('Total number of words:', n_vocab)

sent_len = 10
dataX, dataY = [], []
for sent in sentences:
    for i in range(1, len(sent)):
        dataX.append([word_to_index[w] for w in sent[:i]])
        dataY.append(word_to_index[sent[i]])
dataX = pad_sequences(dataX, sent_len)
n_patterns = len(dataX)
print('Total Patterns:', n_patterns)

X = np.reshape(dataX, (n_patterns, sent_len, 1))
X = X/float(n_vocab)
y = np_utils.to_categorical(dataY)

# 2 layer stacked LSTM model
model = Sequential()
model.add(LSTM(1024, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(1024))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


# define the checkpoint
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# training (fit the model)
model.fit(X, y, epochs=50, batch_size=64, callbacks=callbacks_list)
