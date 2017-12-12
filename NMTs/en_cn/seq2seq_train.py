from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense
from keras.layers import Embedding
import numpy as np
import jieba
import re, pickle


# In[8]:

batch_size = 64
epochs = 100
latent_dim = 256
num_samples = 10000

BOS, EOS = '😀', '🔴'


# ## Loading data

# In[9]:

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
        cn_tokens = [BOS]+list(jieba.cut(output_text))+[EOS]
        cn_vocs.update(cn_tokens)
        input_texts.append(en_words)
        output_texts.append(cn_tokens)
        cnt += 1
        if cnt >= num_samples:
            break


# ## Generate word-int mapping

# In[10]:

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

# In[11]:

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

'''
for i in range(2):
    print('Input texts = %s' % (' '.join(input_texts[i])))
    print('Input int = %s' % int_input_data[i])
    print('Padded input = %s' % encoder_input_data[i])
    
    print('Output texts = %s' % (''.join(output_texts[i])))
    print('Output int = %s' % (int_output_data[i]))
    print('Padded output = %s' % (decoder_input_data[i]))
'''

# ## Load Pre-trained embedding

# In[12]:

with open('en_to_vec.pickle', 'rb') as handle:
    en_to_vec = pickle.load(handle)

embedding_matrix_en = np.zeros((num_encoder_tokens, latent_dim))
for word, i in en_to_int.items():
    embedding_vector = en_to_vec.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix_en[i] = embedding_vector
        
with open('cn_to_vec.pickle', 'rb') as handle:
    cn_to_vec = pickle.load(handle)

embedding_matrix_cn = np.zeros((num_decoder_tokens, latent_dim))
for word, i in cn_to_int.items():
    embedding_vector = cn_to_vec.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix_cn[i] = embedding_vector   


# ## Define embedding layer

# In[13]:

embedding_layer_en = Embedding(input_dim=num_encoder_tokens,
                            output_dim=latent_dim,
                            weights=[embedding_matrix_en],
                            input_length=max_encoder_seq_length,
                            trainable=False)

embedding_layer_cn = Embedding(input_dim=num_decoder_tokens,
                            output_dim=latent_dim,
                            weights=[embedding_matrix_cn],
                            input_length=max_decoder_seq_length,
                            trainable=False)


# ## Define the model

# In[ ]:

# Define the input sequence
encoder_inputs = Input(shape=(None,))
y = embedding_layer_en(encoder_inputs)
_, state_h, state_c = LSTM(latent_dim, return_state=True)(y)
encoder_states = [state_h, state_c]

# Set up the decoder and use encoder_states as initial state
decoder_inputs = Input(shape=(None,))
y = embedding_layer_cn(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True,return_state=True)
x, _, _ = decoder_lstm(y, initial_state=encoder_states)
decoder_outputs = Dense(num_decoder_tokens, activation='softmax')(x)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size, epochs=epochs,validation_split=0.2)
model.save('s2s.h5')