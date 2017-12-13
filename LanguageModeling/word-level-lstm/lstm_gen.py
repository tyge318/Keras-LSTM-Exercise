
# coding: utf-8

# In[1]:

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
import pickle


# In[2]:

raw_texts = open("wonderland.txt").read()
raw_texts = raw_texts.strip().lower()

sentences = list(map(lambda x: nltk.word_tokenize(x), sent_tokenize(raw_texts)))
sentences = [['@']+sent+['#'] for sent in sentences]


# In[3]:

words = [word for sentence in sentences for word in sentence]
words = sorted(list(set(words)))
word_to_index = {w: i+1 for i, w in enumerate(words)}
index_to_word = {i+1: w for i, w in enumerate(words)}

n_vocab = len(words)+1
print('Total number of words:', n_vocab)
with open('word_mappings.pickle', 'wb') as f:
    pickle.dump([word_to_index, index_to_word], f, protocol=pickle.HIGHEST_PROTOCOL)


# In[4]:

sent_len = 10
dataX, dataY = [], []
for sent in sentences:
    for i in range(1, len(sent)):
        dataX.append([word_to_index[w] for w in sent[:i]])
        dataY.append(word_to_index[sent[i]])
dataX = pad_sequences(dataX, sent_len)
n_patterns = len(dataX)
print('Total Patterns:', n_patterns)


# In[5]:

X = np.reshape(dataX, (n_patterns, sent_len, 1))
X = X/float(n_vocab)
y = np_utils.to_categorical(dataY)


# In[6]:

# 2 layer stacked LSTM model
model = Sequential()
model.add(LSTM(1024, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(1024))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

filename = "weights-improvement-50-1.4122.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

print(y.shape)


# In[20]:

# pick a random seed
if type(dataX) != list:
    dataX = dataX.tolist()
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:",' '.join([index_to_word.get(value, '') for value in pattern]))
# generate characters
for i in range(1000):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = index_to_word[index]
    if result == '#':
        start = np.random.randint(0, len(dataX)-1)
        pattern = dataX[start]
        continue
    sys.stdout.write(' '+result)
    pattern.append(index)
    pattern = pattern[1:]
print("\nDone.")

