
# coding: utf-8

# In[1]:

from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense
from keras.layers.embeddings import Embedding
import numpy as np
import jieba
import re


# In[2]:

batch_size = 64
epochs = 15
latent_dim = 256
num_samples = 10000


# ## Loading data

# In[3]:

input_texts, output_texts = [], []
en_vocs, cn_vocs = set(), set()
pattern = re.compile('[\W_]+')
with open('cmn.txt', 'r') as f:
    cnt = 0
    for line in f:
        # source and target are seperated by tab; also lower the letters
        input_text, output_text = line.lower().split('\t')
        # remove English punctuations
        en_words = list(map(lambda x: pattern.sub('', x), input_text.split()))
        en_vocs.update(en_words)
        # remove Chinese punctuations
        output_text = re.sub( "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。?？、~@#￥%……&*（ ）]+", '',output_text)  
        cn_tokens = ['\t']+list(jieba.cut(output_text))+['\n']
        cn_vocs.update(cn_tokens)
        input_texts.append(en_words)
        output_texts.append(cn_tokens)
        cnt += 1
        if cnt >= num_samples:
            break


# ## Generate word-int mapping

# In[4]:

en_vocs = sorted(list(en_vocs))
cn_vocs = sorted(list(cn_vocs))
en_to_int, int_to_en = {w: i+1 for i, w in enumerate(en_vocs)}, {i+1: w for i, w in enumerate(en_vocs)}
cn_to_int, int_to_cn = {w: i+1 for i, w in enumerate(cn_vocs)}, {i+1: w for i, w in enumerate(cn_vocs)}

num_encoder_tokens = len(en_vocs)+1
num_decoder_tokens = len(cn_vocs)+1
max_encoder_seq_length = max(map(len, input_texts))
max_decoder_seq_length = max(map(len, output_texts))

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)


# ## Vectorization and Padding

# In[5]:

int_input_data = [[en_to_int[w] for w in row] for row in input_texts]
int_output_data = [[cn_to_int[w] for w in row] for row in output_texts]
encoder_input_data = pad_sequences(int_input_data, maxlen=max_encoder_seq_length, padding='post')
decoder_input_data = pad_sequences(int_output_data, maxlen=max_decoder_seq_length, padding='post')
#decoder_target_data = pad_sequences(decoder_input_data[:,1:], maxlen=max_decoder_seq_length, padding='post')

#decoder_target_data needs to be one-hot encoded
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, row in enumerate(int_output_data):
    for t, w in enumerate(row):
        if t == 0:
            continue
        decoder_target_data[i, t-1, w] = 1
#print(decoder_target_data.shape)


# ## Define the model

# In[ ]:

# Define the input sequence
encoder_inputs = Input(shape=(None,))
encoder = LSTM(latent_dim, return_state=True)
embedding_inputs = Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
_, state_h, state_c = encoder(embedding_inputs)
encoder_states = [state_h, state_c]

# Set up the decoder and use encoder_states as initial state
decoder_inputs = Input(shape=(None,))
embedding_inputs = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True,return_state=True)
x, _, _ = decoder_lstm(embedding_inputs, initial_state=encoder_states)
decoder_outputs = Dense(num_decoder_tokens, activation='softmax')(x)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size, epochs=epochs,validation_split=0.2)

'''
model.save('s2s.h5')

model = load_model('s2s.h5')
'''
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
embedding_inputs = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)

x, state_h, state_c = decoder_lstm(embedding_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = Dense(num_decoder_tokens, activation='softmax')(x)
decoder_model = Model([decoder_inputs]+decoder_states_inputs, [decoder_outputs]+decoder_states)

def decode_sequence(input_sentence):
    # Encode the input as state vectors.
    ##### input_seq = Embedding(num_encoder_tokens, latent_dim)(input_sentence)
    ##### print('@@@@@@@@@@@@@@@@@@@@@@@@@',type(input_seq))
    states_value = encoder_model.predict(input_sentence)
    
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = cn_to_int['\t']
    
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        ###########embedding_inputs = Embedding(num_decoder_tokens, latent_dim)(target_seq)
        output_tokens, h, c = decoder_model.predict([target_seq]+states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = int_to_cn[sampled_token_index]
        decoded_sentence += sampled_word

        if (sampled_word == '\n' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1)
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]
    
    return decoded_sentence

import random
random_picked = [random.randint(0, 9999) for _ in range(10)]

for i in random_picked:
    input_sent = encoder_input_data[i]
    decoded_sentence = decode_sequence(input_sent)
    print('-')
    print('Input sentence:', ' '.join(input_texts[i]))
    print('Decoded sentence:', decoded_sentence)

