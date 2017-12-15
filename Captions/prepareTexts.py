import string

def load_doc(filename):
    with open(filename, 'r') as f:
        text = f.read()
    return text

def load_descriptions(doc):
    mapping = {}
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        image_id, text = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        text = ' '.join(text)
        mapping.setdefault(image_id, []).append(text)
    return mapping

def clean_description(texts):
    puncts = str.maketrans('', '', string.punctuation)
    vocabs = set()
    for key, text_list in texts.items():
        for i, cur in enumerate(text_list):
            tokens = cur.split()
            tokens = [w.lower() for w in tokens]
            tokens = [w.translate(puncts) for w in tokens]
            # remove haning 's' and 'a'
            tokens = [w for w in tokens if len(w)>1]
            tokens = [w for w in tokens if w.isalpha()]
            vocabs.update(tokens)
            text_list[i] = ' '.join(tokens)
    return list(vocabs)

def save_texts(texts, filename):
    with open(filename, 'w') as f:
        for key, text_list in texts.items():
            for text in text_list:
                f.write(key + ' ' + text + '\n')

filename = 'Flickr8k_text/Flickr8k.token.txt'
doc = load_doc(filename)
texts = load_descriptions(doc)
print('Loaded: %d' % len(texts))
vocabulary = clean_description(texts)
print('Vocabulary size = %d' % len(vocabulary))
save_texts(texts, 'descriptions.txt')
