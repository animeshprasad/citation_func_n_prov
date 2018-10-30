#__author__ = "Xu Suan, and Animesh Prasad"
#__copyright__ = "Copyright 2018, WING-NUS"
#__email__ = "animesh@u.nus.edu"
"""
Experiments for MTL vs baselines on golden data.
"""
from __future__ import division, print_function

import os, csv
import numpy as np

import config.config as config
import data.data as data
import sklearn.metrics as metrics
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.utils import class_weight


from keras.utils import np_utils
from keras.engine.topology import Input
from keras.layers import Conv1D, Dense, \
    GlobalMaxPooling1D, Merge, Embedding, Dropout, Masking
from keras.layers.convolutional import Convolution1D
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf


"""
Seed for Replicablity.
"""
seed = 1027
np.random.seed(seed)
tf.set_random_seed(seed)

MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 50
GLOVE_DIR = config.GLOVE_DIR
GLOVE_FILE = 'glove.6B.100d.txt' # Note glove vector dimension should be same as that of embedding
EMBEDDING_DIM = 100
NO_FOLDS = 2


NB_FILTER = 128 # When filter size is 256, both are better
BATCH_SIZE = 32
EPOCH = 20

"""
Data reading and saving from disk (so that data processing is done only once).
"""
directory = config.DATA_DIR
datafiles = config.DATA_FILES['prov']

provs_index = {'Prov': 0, 'Non-Prov': 1}
funcs_index = {'Neut': 3, 'Pos': 2, 'CoCo': 1, 'Weak': 0}

# Provenance dataset start
golden_train = data.read_json_data(datafiles['golden_train'])
golden_test = data.read_json_data(datafiles['golden_test'])

dataset = filter(lambda x: x['function'] != 'Error', golden_train + golden_test)
dataset_pos = filter(lambda x: x['label'] == 'Prov', dataset)
dataset_neg = filter(lambda x: x['label'] == 'Non-Prov', dataset)

print ('Provenance data has %d positive and %d negative samples' % (len(dataset_pos), len(dataset_neg)))
# Provenance dataset end

# Function dataset start
datafiles = config.DATA_FILES['func']
golden_train = data.read_json_data(datafiles['golden_train'])
golden_test = data.read_json_data(datafiles['golden_test'])

dataset_func = filter(lambda d: d['label'] != 'Error',
                      golden_train + golden_test)

print ('Function data has %d samples' %  len(dataset_func))



texts = map(lambda d: d['context'][1], dataset_func)
texts_a_pos = map(lambda d: d['context'][1], dataset_pos)
texts_b_pos = map(lambda d: reduce(lambda x, y: x + ' ' + y, d['provenance']),
                  dataset_pos)
texts_a_neg = map(lambda d: d['context'][1], dataset_neg)
texts_b_neg = map(lambda d: reduce(lambda x, y: x + ' ' + y, d['provenance']),
                  dataset_neg)
texts += texts_a_pos + texts_b_pos + texts_a_neg + texts_b_neg

ys = map(lambda d: funcs_index[d['label']], dataset_func)

citing2id = {}
id2neg = {}

for index, instance in enumerate(dataset_pos):
    # Current sentence + citing paper id
    # Used to group pos and neg instances together
    key = (instance['context'][1], instance['citing'])
    citing2id[key] = index

for index, instance in enumerate(dataset_neg):
    key = (instance['context'][1], instance['citing'])
    key_id = citing2id[key]
    if key_id not in id2neg:
        id2neg[key_id] = []
    # id2neg: index of the pos instance to corresponding neg instances ids
    id2neg[key_id].append(index)

funcs_pos = map(lambda d: funcs_index[d['function']], dataset_pos)
funcs_pos = np_utils.to_categorical(np.asarray(funcs_pos))

print('Found %s texts.' % len(texts))

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)

sequences_a_pos = tokenizer.texts_to_sequences(texts_a_pos)
sequences_b_pos = tokenizer.texts_to_sequences(texts_b_pos)
sequences_a_neg = tokenizer.texts_to_sequences(texts_a_neg)
sequences_b_neg = tokenizer.texts_to_sequences(texts_b_neg)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

xs_a_pos = pad_sequences(sequences_a_pos, maxlen=MAX_SEQUENCE_LENGTH)
xs_b_pos = pad_sequences(sequences_b_pos, maxlen=MAX_SEQUENCE_LENGTH)
xs_a_neg = pad_sequences(sequences_a_neg, maxlen=MAX_SEQUENCE_LENGTH)
xs_b_neg = pad_sequences(sequences_b_neg, maxlen=MAX_SEQUENCE_LENGTH)

kf = KFold(n_splits=NO_FOLDS)

y_pred_func_all = []
y_test_func_all = []
y_pred_prov_all = []
y_test_prov_all = []
y_pred_only_func_all = []
y_test_only_func_all = []
y_pred_only_prov_all = []
y_test_only_prov_all = []

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, GLOVE_FILE))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


texts = map(lambda d: d['context'][1], dataset_func)
sequences = tokenizer.texts_to_sequences(texts)
xs = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
ys = np_utils.to_categorical(np.asarray(ys))


# --------------------------------------------------------------------------------------

with open('outfile_f.csv', 'wb') as f1, open('outfile_p.csv', 'wb') as f2:
    csv_writer_f = csv.writer(f1, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer_f.writerow(['Citing', 'Target', 'Actual', 'Baseline', 'MTL'])
    csv_writer_p = csv.writer(f2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer_p.writerow(['Citance', 'Actual', 'Baseline', 'MTL'])

    # split the data into a training set and a test set for k-fold reporting
    for func_split, prov_split in zip(kf.split(xs), kf.split(xs_a_pos)):
        x_train_func = [xs[j] for j in func_split[0]]
        y_train_func = [ys[j] for j in func_split[0]]
        x_train_a = [xs_a_pos[j] for j in prov_split[0]]
        x_train_b = [xs_b_pos[j] for j in prov_split[0]]
        y_train_prov = [[1, 0] for j in prov_split[0]]

        for j in prov_split[0]:
            negs = id2neg[j] if j in id2neg else []
            for neg in negs:
                x_train_a.append(xs_a_neg[neg])
                x_train_b.append(xs_b_neg[neg])
                y_train_prov.append([0, 1])

        x_train_a = np.array(x_train_a)
        x_train_b = np.array(x_train_b)
        y_train_prov = np.array(y_train_prov)
        x_train_func = np.array(x_train_func)
        y_train_func = np.array(y_train_func)


        x_test_func = [xs[j] for j in func_split[1]]
        y_test_func = [ys[j] for j in func_split[1]]
        x_test_a = [xs_a_pos[j] for j in prov_split[1]]
        x_test_b = [xs_b_pos[j] for j in prov_split[1]]
        y_test_prov = [[1, 0] for j in prov_split[1]]

        for j in prov_split[1]:
            negs = id2neg[j] if j in id2neg else []
            for neg in negs:
                x_test_a.append(xs_a_pos[j])
                x_test_b.append(xs_b_neg[neg])
                y_test_prov.append([0, 1])

        x_test_a = np.array(x_test_a)
        x_test_b = np.array(x_test_b)
        x_test_func = np.array(x_test_func)
        y_test_func = np.array(y_test_func)
        y_test_prov = np.array(y_test_prov)

        y_train_copy = map(lambda x: x.tolist().index(1), y_train_func)
        weight_f = class_weight.compute_class_weight('balanced', np.unique(y_train_copy), y_train_copy)
        print('Applying function class weight' % weight_f)

        y_train_copy = map(lambda x: x.tolist().index(1), y_train_prov)
        weight_p = class_weight.compute_class_weight('balanced', np.unique(y_train_copy), y_train_copy)
        print('Applying function class weight' % weight_p)

        #print (x_test_a.shape, x_test_b.shape, x_test_func.shape, y_test_func.shape, y_test_prov.shape)
        #print (x_train_a.shape, x_train_b.shape, x_train_func.shape, y_train_func.shape, y_train_prov.shape)

        # ---------- Start of MTL ----------

        embedding_layer1 = Embedding(len(word_index) + 1,
                                     EMBEDDING_DIM,
                                     weights=[embedding_matrix],
                                     input_length=MAX_SEQUENCE_LENGTH, trainable=False)

        sequence_input1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        mask1 = Masking(mask_value=0)(sequence_input1)
        embedded_sequences1 = embedding_layer1(mask1)
        x1 = Conv1D(NB_FILTER, 5, activation='relu')(embedded_sequences1)
        xA = GlobalMaxPooling1D()(x1)
        x1 = Dropout(Dense(32, activation='relu'))(xA)

        embedding_layer2 = Embedding(len(word_index) + 1,
                                     EMBEDDING_DIM,
                                     weights=[embedding_matrix],
                                     input_length=MAX_SEQUENCE_LENGTH, trainable=False)

        sequence_input2 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        mask2 = Masking(mask_value=0)(sequence_input2)
        embedded_sequences2 = embedding_layer2(mask2)
        x2 = Conv1D(NB_FILTER, 5, activation='relu')(embedded_sequences2)
        x2 = GlobalMaxPooling1D()(x2)
        x2 = Dropout(Dense(32, activation='relu'))(x2)

        x = Merge(mode='mul')([x1, x2])
        x = Merge(mode='concat')([x1, x2, x])
        preds_p = Dense(2, activation='softmax')(x)

        model2 = Model([sequence_input1, sequence_input2], preds_p)

        x = Dropout(Dense(32, activation='relu'))(xA)
        preds_f = Dense(len(funcs_index), activation='softmax')(x)

        model1 = Model(sequence_input1, preds_f)


        model2.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])


        model1.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop',
                       metrics=['acc'])

        print (model1.summary())
        print (model2.summary())

        count = 0
        while count < EPOCH:
            model2.fit([x_train_a, x_train_b], y_train_prov,
                  nb_epoch=1, batch_size=BATCH_SIZE, class_weight=weight_p)
            model1.fit(x_train_func, y_train_func,
                   nb_epoch=1, batch_size=BATCH_SIZE, class_weight=weight_f)
            count += 1

        y_pred_func = model1.predict(
            x_test_func)

        y_pred_prov = model2.predict(
            [x_test_a, x_test_b])

        y_pred_func = map(lambda x: pd.Series(x).idxmax(), y_pred_func)
        y_pred_prov = map(lambda x: pd.Series(x).idxmax(), y_pred_prov)

        y_test_func = data.compress_y(y_test_func)
        y_test_prov = data.compress_y(y_test_prov)

        # Generate classification report
        print ('MTL_Func Classification Report')
        print (metrics.classification_report(y_test_func, y_pred_func, digits=4))
        print ('MTL_Prov Classification Report')
        print (metrics.classification_report(y_test_prov, y_pred_prov, digits=4))

        y_pred_func_all += y_pred_func
        y_test_func_all += y_test_func
        y_pred_prov_all += y_pred_prov
        y_test_prov_all += y_test_prov

        # ---------- End of MTL ----------

        # ---------- Start of citation function ----------


        embedding_layer = Embedding(len(word_index) + 1,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH, trainable=False)

        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        mask = Masking(mask_value=0)(sequence_input)
        embedded_sequences = embedding_layer(mask)
        x = Convolution1D(nb_filter=NB_FILTER,
                         filter_length=5,
                         activation='relu')(embedded_sequences)
        x = GlobalMaxPooling1D()(x)
        x = Dropout(Dense(NB_FILTER, activation='relu'))(x)
        preds = Dense(len(funcs_index), activation='softmax')(x)

        model = Model(sequence_input, preds)

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])
        print(model.summary())

        model.fit(x_train_func, y_train_func,
                  nb_epoch=EPOCH, batch_size=BATCH_SIZE, class_weight=weight_f)

        y_pred_probs = model.predict(x_test_func)

        y_pred_func = map(lambda x: pd.Series(x).idxmax(), y_pred_probs)

        y_test_func = data.compress_y(y_test_func)

        # Generate classification report
        y_pred_only_func_all += y_pred_func
        y_test_only_func_all += y_test_func

        # ---------- End of citation function ----------

        # ---------- Start of citation provenance ----------

        embedding_layer1 = Embedding(len(word_index) + 1,
                                     EMBEDDING_DIM,
                                     weights=[embedding_matrix],
                                     input_length=MAX_SEQUENCE_LENGTH, trainable=False)

        sequence_input1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        mask1 = Masking(mask_value=0)(sequence_input1)
        embedded_sequences1 = embedding_layer1(mask1)
        x1 = Conv1D(NB_FILTER, 5, activation='relu')(embedded_sequences1)
        x1 = GlobalMaxPooling1D()(x1)
        x1 = Dropout(Dense(32, activation='relu'))(x1)

        embedding_layer2 = Embedding(len(word_index) + 1,
                                     EMBEDDING_DIM,
                                     weights=[embedding_matrix],
                                     input_length=MAX_SEQUENCE_LENGTH)

        sequence_input2 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        mask2 = Masking(mask_value=0)(sequence_input2)
        embedded_sequences2 = embedding_layer2(mask2)
        x2 = Conv1D(NB_FILTER, 5, activation='relu')(embedded_sequences2)
        x2 = GlobalMaxPooling1D()(x2)
        x2 = Dropout(Dense(32, activation='relu'))(x2)

        x = Merge(mode='mul')([x1, x2])
        x = Merge(mode='concat')([x1, x2, x])
        preds_p = Dense(2, activation='softmax')(x)

        model = Model([sequence_input1, sequence_input2], preds_p)

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])
        print (model.summary())

        model.fit([x_train_a, x_train_b], y_train_prov,
                  nb_epoch=EPOCH, batch_size=BATCH_SIZE, class_weight=weight_p)


        y_pred_probs = model.predict([x_test_a, x_test_b])

        y_pred_prov = map(lambda x: pd.Series(x).idxmax(), y_pred_probs)

        y_test_prov = data.compress_y(y_test_prov)

        # Generate classification report
        y_pred_only_prov_all += y_pred_prov
        y_test_only_prov_all += y_test_prov

        # ---------- End of citation provenance ----------


    print ('MTL_Func')
    print (metrics.classification_report(y_test_func_all, y_pred_func_all, digits=4))
    print ('MTL_Prov')
    print (metrics.classification_report(y_test_prov_all, y_pred_prov_all, digits=4))
    print ('Plain_Func')
    print (metrics.classification_report(y_test_only_func_all, y_pred_only_func_all, digits=4))
    print ('Plain_Prov')
    print (metrics.classification_report(y_test_only_prov_all, y_pred_only_prov_all, digits=4))

