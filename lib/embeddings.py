# References:
# https://github.com/danielfrg/word2vec
# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

import nltk
import os
import random
import numpy as np
import config.config as config
import config.params_func as params_func
import config.params_prov as params_prov
from nltk.corpus import stopwords

embeddings = {}
stop_words = []


def init_embeddings(mode):
    global embeddings
    if mode == 'func':
        embedding_dims = params_func.embedding_dims
    else:
        embedding_dims = params_prov.embedding_dims

    embeddings = {}
    filename = 'glove.6B.' + str(embedding_dims) + 'd.txt'
    f = open(os.path.join(config.GLOVE_DIR, filename))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings[word] = coefs
    print('Found %s word vectors.' % len(embeddings))


def init_stopwords():
    global stop_words
    stop_words = set(stopwords.words('english'))
    stop_words.update(
        ['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])


def init(mode):
    init_embeddings(mode)
    init_stopwords()


def is_word(word):
    return word in embeddings


def get_random_vector(dim):
    """
    Generate a random vector for OOV words. The elements have abs values < 2.
    """
    vector = [0] * dim
    for i in range(0, dim):
        vector[i] = random.uniform(-2, 2)
    return vector


def get_embedding_matrix(word_index, mode):
    """
    Return the embedding matrix where the i-th column represents the word
    whose index is i.
    - word_index: A dictionary mapping words to their corresponding indices
    """
    if mode == 'func':
        nb_words = params_func.nb_words
        embedding_dims = params_func.embedding_dims
    else:
        nb_words = params_prov.nb_words
        embedding_dims = params_prov.embedding_dims

    embedding_matrix = np.zeros((nb_words, embedding_dims))

    # Random vectors for start symbol and OOV words
    embedding_matrix[0] = get_random_vector(embedding_dims)
    embedding_matrix[1] = get_random_vector(embedding_dims)
    embedding_matrix[2] = get_random_vector(embedding_dims)

    for word, i in word_index.items():
        if i >= nb_words:
            continue

        if is_word(word):
            embedding_vector = embeddings.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        else:
            # OOV words: generate a random vector
            embedding_matrix[i] = get_random_vector(embedding_dims)

    return embedding_matrix


"""
- TODO The following functions are not used, can be removed
"""


def default_sent2vec(sentence):
    """
    Given a sentence string, tokenize it using NLTK, remove stopwords, and
    output its vector representation.
    """
    words = nltk.word_tokenize(sentence)
    words = filter(lambda (w): w not in stop_words, words)
    return default_words2sent(words)


def default_words2sent(words):
    weights = [1] * len(words)
    return words2sent(words, weights)


def words2sent(words, weights):
    """
    Given a list of words and their respective weights, compute the sentence
    vector. Assume that the words are all valid.
    """
    sent_vector = np.zeros((1, params.embedding_dims))
    weights_sum = sum(weights)

    for i, w in enumerate(words):
        if not is_word(w):
            continue
        else:
            sent_vector += embeddings[w] * weights[i]
    sent_vector /= weights_sum

    return sent_vector
