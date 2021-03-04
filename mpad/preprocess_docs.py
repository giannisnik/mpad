import re
import os
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from collections import Counter, OrderedDict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_word2vec(fname, word2idx):
    word_vecs = np.zeros((len(word2idx) + 1, 300))
    unknown_words = set()
    model = KeyedVectors.load_word2vec_format(fname, binary=True)
    for word in word2idx.keys(): # could also be done with enumerate
        if word in model:
            word_vecs[word2idx[word], :] = model[word]
        else:
            unknown_words.add(word)
            word_vecs[word2idx[word], :] = np.random.uniform(-0.25, 0.25, 300)
    print("Existing vectors:", len(word2idx) - len(unknown_words))
    return word_vecs

def clean_str(s):
    s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)
    s = re.sub(r"\'s", " \'s", s)
    s = re.sub(r"\'ve", " \'ve", s)
    s = re.sub(r"n\'t", " n\'t", s)
    s = re.sub(r"\'re", " \'re", s)
    s = re.sub(r"\'d", " \'d", s)
    s = re.sub(r"\'ll", " \'ll", s)
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\(", " \( ", s)
    s = re.sub(r"\)", " \) ", s)
    s = re.sub(r"\?", " \? ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip().lower().split()


def load_file(filename):
    """
    Loads a tab separated file with a single label
    """
    labels = []
    docs = []

    with open(filename, encoding='utf8', errors='ignore') as f:
        for line in f:
            content = line.split('\t')
            labels.append(content[0])
            docs.append(content[1][:-1])

    return docs, labels

def encode_multi_class_labels(labels):
    """
    Converts raw label(description)s to one-hot-encoding
    """
    enc = LabelEncoder()
    labels = enc.fit_transform(labels)

    nclass = len(enc.classes_)
    y = list()
    for i in range(len(labels)):
        t = np.zeros(1)
        t[0] = labels[i]
        y.append(t)

    return y, nclass

def multi_class_train_test_split(X, y, test_size):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = test_size, random_state = 42)

    return X_train, X_test, y_train, y_test



class CorpusPreProcessor():
    def __init__(self, min_freq_word = 1, multi_label=False):

        self.docs = []
        self.labels = []
        self.word2idx = OrderedDict()
        self.min_freq_word = min_freq_word
        self.multi_label = multi_label

    def load_clean_corpus(self, in_path):

        # Load in the raw docs
        docs, labels = load_file(in_path)

        # Clean the docs
        docs = [self.clean_doc(doc) for doc in docs]

        # Extract the vocab
        word_counter = Counter([word for doc in docs for word in doc])
        for ix, (word, count) in enumerate(word_counter.most_common()):
            if count >= self.min_freq_word:
                self.word2idx[word] = ix
            else:
                break

        # Convert labels
        labels, n_labels = self.process_labels(labels)

        return docs, labels, n_labels, self.word2idx

    def split_corpus(self, docs, labels, test_size):

        if not self.multi_label:
            # Multi-class train/test split
            return multi_class_train_test_split(docs, labels, test_size=test_size)
        else:
            raise NotImplementedError()

    def process_labels(self, labels):
        # First only implement for multi-class classification
        if not self.multi_label:
            y, n_labels = encode_multi_class_labels(labels)
        else:
            raise NotImplementedError()

        return y, n_labels




    def clean_doc(self, doc):
        return clean_str(doc)

    def get_vocab(self):
        pass

    def load_embeddings(self, f_path, vocab, embedding_type='word2vec'):

        if embedding_type == 'word2vec':
            embeddings = load_word2vec(f_path, vocab)

        return embeddings

