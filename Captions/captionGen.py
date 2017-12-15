from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model, load_model
import sys

def extract_features(filename):
    model = VGG16()

    # Removing the last classification layer to ge the features
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

    image = load_img(filename, target_size = (224, 224))
        
    image = img_to_array(image)
    # RGB has 3 channels of 224x224
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    feature = model.predict(image, verbose=0)
    return feature

# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'bos'
    ans = []
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None or word == 'eos':
            break
        in_text += ' ' + word
        ans += [word]
    return ' '.join(ans)

tokenizer = load(open('tokenizer.pkl', 'rb'))
max_length = 34
model = load_model('model-ep032-loss3.450-val_loss3.916.h5')
photo = extract_features(sys.argv[1])
description = generate_desc(model, tokenizer, photo, max_length)
print(description)

