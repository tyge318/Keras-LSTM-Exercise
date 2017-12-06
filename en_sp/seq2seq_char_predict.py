
# coding: utf-8

# In[1]:

from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense
from keras.layers.embeddings import Embedding
import numpy as np
import re


# In[2]:

batch_size = 64
epochs = 100
latent_dim = 256
num_samples = 10000


# ## Loading data

# In[3]:

input_texts, output_texts = [], []
en_vocs, sp_vocs = set(), set()
with open('spa.txt', 'r') as f:
    cnt = 0
    for line in f:
        # source and target are seperated by tab; also lower the letters
        input_text, output_text = line.lower().split('\t')
        input_texts.append(input_text)
        output_text = '\t' + output_text + '\n'
        output_texts.append(output_text)
        en_vocs.update(list(input_text))
        sp_vocs.update(list(output_text))
        cnt += 1
        if cnt >= num_samples:
            break


# ## Generate word-int mapping

# In[4]:

en_vocs = sorted(list(en_vocs))
sp_vocs = sorted(list(sp_vocs))
en_to_int, int_to_en = {w: i for i, w in enumerate(en_vocs)}, {i: w for i, w in enumerate(en_vocs)}
sp_to_int, int_to_sp = {w: i for i, w in enumerate(sp_vocs)}, {i: w for i, w in enumerate(sp_vocs)}

num_encoder_tokens = len(en_vocs)
num_decoder_tokens = len(sp_vocs)
max_encoder_seq_length = max(map(len, input_texts))
max_decoder_seq_length = max(map(len, output_texts))

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)


# ## Vectorization and Padding

# In[5]:

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, en_to_int[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, sp_to_int[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, sp_to_int[char]] = 1.

# ## Define the model

# In[ ]:

# Define the input sequence
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# Set up the decoder and use encoder_states as initial state
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True,return_state=True)
x, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_outputs = Dense(num_decoder_tokens, activation='softmax')(x)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size, epochs=epochs,validation_split=0.2)

model.save('s2s.h5')

#model = load_model('s2s.h5')

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

x, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = Dense(num_decoder_tokens, activation='softmax')(x)
decoder_model = Model([decoder_inputs]+decoder_states_inputs, [decoder_outputs]+decoder_states)

def decode_sequence(input_sentence):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_sentence)
    
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.
    
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq]+states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = int_to_sp[sampled_token_index]
        decoded_sentence += [sampled_word]

        if (sampled_word == '\n' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1)
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]
    
    return ' '.join(decoded_sentence)

import random
random_picked = [random.randint(0, 9999) for _ in range(10)]

for i in random_picked:
    input_sent = encoder_input_data[i]
    decoded_sentence = decode_sequence(input_sent)
    print('-')
    print('Input sentence:', ' '.join(input_texts[i]))
    print('Decoded sentence:', decoded_sentence)

