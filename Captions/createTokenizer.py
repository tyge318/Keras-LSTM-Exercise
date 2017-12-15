from keras.preprocessing.text import Tokenizer
import pickle

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

# load training dataset (6K)
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
# save the tokenizer
pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))
