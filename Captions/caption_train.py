from numpy import array
from os import listdir
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import preprocess_input

def load_doc(filename):
    with open(filename, 'r') as f:
        text = f.read()
    return text

def load_set(filename):
    doc = load_doc(filename)
    dataset = set()
    for line in doc.split('\n'):
        if len(line) < 1:
            continue
        key = line.split('.')[0]
        dataset.add(key)
    return dataset

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
    doc = load_doc(filename)
    texts = {}
    for line in doc.split('\n'):
        if len(line) == 0:
            continue
        tokens = line.split()
        image_id, text = tokens[0], tokens[1:]
        if image_id in dataset:
            desc = 'BOS ' + ' '.join(text) + ' EOS'
            texts.setdefault(image_id, []).append(desc)
    return texts

#load photo features
def load_photo_features(filename, dataset):
    with open(filename, 'rb') as f:
        all_features = load(f)
    features = {key: all_features[key] for key in dataset}
    return features

# convert a dictionary of clean descriptions to list
def to_lines(descriptions):
    all_desc = []
    for key in descriptions.keys():
        all_desc += descriptions[key]
    return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(map(lambda d: len(d.split()), lines))

def create_sequences(tokenizer, max_length, descriptions, photos):
    X1, X2, y = list(), list(), list()
    # walk through each image identifier
    for key, desc_list in descriptions.items():
        # walk through each description for the image
        for desc in desc_list:
            # encode the sequence
            seq = tokenizer.texts_to_sequences([desc])[0]
            # split one sequence into multiple X,y pairs
            for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                # encode output sequence
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                # store
                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)
    return array(X1), array(X2), array(y)

def create_sequences_single(tokenizer, max_length, desc_list, photo):
    X1, X2, y = [], [], []
    vocab_size = len(tokenizer.word_index)+1
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen = max_length)[0]
            out_seq = to_categorical([out_seq], num_classes = vocab_size)[0]
            X1.append(photo[0])
            X2.append(in_seq)
            y.append(out_seq)
    return [array(X1), array(X2), array(y)]

def define_model(vocab_size, max_length, filename = None):
    # feature extractor model
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    if filename:
        model = load_model(filename)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    # summarize model
    # print(model.summary())
    # plot_model(model, to_file='model.png', show_shapes=True)
    return model

# train dataset

# load training dataset (6K)
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
train_features = load_photo_features('features.pkl', train)
print('Photos: train=%d' % len(train_features))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index)+1
print('Vocabulary Size: %d' % vocab_size)

# determine max sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)
'''
# prepare sequences
X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features)
'''
def data_generator(descriptions, tokenizer, max_length):
    directory = 'Flicker8k_Dataset'
    while True:
        for image_id in train:
            image = train_features[image_id]
            desc = descriptions[image_id]
            in_img, in_seq, out_word = create_sequences_single(tokenizer, max_length, desc, image)
            yield[[in_img, in_seq], out_word]
# dev dataset

# load test set
filename = 'Flickr8k_text/Flickr_8k.devImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = load_photo_features('features.pkl', test)
print('Photos: test=%d' % len(test_features))
# prepare sequences
X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features)

# fit model

model = define_model(vocab_size, max_length, 'model-ep010-loss3.900-val_loss3.950.h5')

filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# fit model
model.fit_generator(data_generator(train_descriptions, tokenizer, max_length), epochs=80, steps_per_epoch=1200, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))







