import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

raw_texts = open("wonderland.txt").read()
raw_texts = raw_texts.lower()

# create mapping of chars to integers and the reverse
chars = sorted(list(set(raw_texts)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(raw_texts)
n_vocab = len(chars)

print "Total Characters: %s" % n_chars
print "Total Vocab: %s" % n_vocab

# generate datasets
seq_length = 100
dataX, dataY = [], []
for i in xrange(n_chars-seq_length):
	seq_in = raw_texts[i:i+seq_length]
	seq_out = raw_texts[i+seq_length]
	dataX.append([char_to_int[c] for c in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print "Total Patterns: %s " % n_patterns

# reshape X to [batch size, time steps, input_dim]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalization
X = X / float(n_vocab)
# one hot encoding
y = np_utils.to_categorical(dataY)

'''
print 'X.shape = %s' % str(X.shape)
print 'y.shape = %s' % str(y.shape)
'''
# LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# training (fit the model)
model.fit(X, y, epochs=50, batch_size=64, callbacks=callbacks_list)






'''
# load the weights
filename = "weights-improvement-20-1.9322.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print 'Seed = %s' % (''.join([int_to_char[i] for i in pattern]) )

#generate characters
output = []
for i in range(1000):
	cur = pattern[-seq_length:]
	x = numpy.reshape(cur, (1, len(cur), 1))
	x = x/float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	output.append(int_to_char[index])
	pattern.append(index)
print ''.join(output)
print '\nDone.'
'''
