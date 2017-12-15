import os
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

def extract_features(path):
    model = VGG16()

    # Removing the last classification layer to ge the features
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

    # summarize
    print(model.summary())

    features = {}
    for fname in os.listdir(path):
        fpath = os.path.join(path, fname)
        image = load_img(fpath, target_size = (224, 224))
        
        image = img_to_array(image)
        # RGB has 3 channels of 224x224
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)

        feature = model.predict(image, verbose=0)
        image_id = fname.split('.')[0]

        features[image_id] = feature

        print('Done one image %s' % fname)
    return features

features = extract_features('Flicker8k_Dataset')
print('Extracted features: %d' % len(features))
with open('features.pkl', 'wb') as f:
    dump(features, f)
