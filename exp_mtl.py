# New file authored 28 Jan 2018
# For citation provenance
# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

"""
Perform the experiments on bootstrapped data and actual annotated data.
"""
import lib.logger, os, sys, random, math
import numpy as np
import pandas as pd
import lib.utils as utils
import lib.file_ops as file_ops
import lib.embeddings as embeddings
import config.config as config
import data.data as data
import data.data_func as data_func
import sklearn.metrics as metrics
from sklearn.model_selection import KFold

# TODO Do not use sentiment features, as they don't seem very helpful
# import data.google as google
import data.mturk as mturk
from sklearn.preprocessing import normalize
from sklearn.utils import class_weight

import keras.backend as K
from keras.utils import np_utils
from keras.engine.topology import Input
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, merge, \
    GlobalMaxPooling1D, Merge, Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.models import Model, Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

"""
Set random seed and fix bug on Dropout usage.
"""
import tensorflow as tf

seed = 1027
np.random.seed(seed)
# tf.python.control_flow_ops = tf
tf.set_random_seed(seed)

MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 50
GLOVE_DIR = config.GLOVE_DIR
EMBEDDING_DIM = 100

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
# Provenance dataset end

# Function dataset start
datafiles = config.DATA_FILES['func']
golden_train = data.read_json_data(datafiles['golden_train'])
golden_test = data.read_json_data(datafiles['golden_test'])

dataset_func = filter(lambda d: d['label'] != 'Error',
                      golden_train + golden_test)
lendiff = len(dataset) - len(dataset_func)
print lendiff
dataset_func += random.sample(dataset_func, lendiff)
# Function dataset end

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

kf = KFold(n_splits=5)

y_pred_func_all = []
y_test_func_all = []
y_pred_prov_all = []
y_test_prov_all = []
y_pred_only_func_all = []
y_test_only_func_all = []
y_pred_only_prov_all = []
y_test_only_prov_all = []

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
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

# -------------------------
texts = map(lambda d: d['context'][1], dataset_func)
sequences = tokenizer.texts_to_sequences(texts)
xs = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
ys = np_utils.to_categorical(np.asarray(ys))

whole_dataset = dataset_func
whole_dataset = dataset_pos + dataset_neg

import csv
with open('/Users/suxuan/Development/func_asdasdsmtl.csv', 'wb') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['Citing', 'Actual', 'MTL'])
    # csv_writer.writerow(['Citing', 'Target', 'Actual', 'MTL'])

    # split the data into a training set and a validation set
    index = -1
    for train_index, test_index in (kf.split(xs_a_pos)):
        index += 1
        x_train_a = [xs_a_pos[j] for j in train_index]
        x_train_b = [xs_b_pos[j] for j in train_index]
        y_train_prov = [[1, 0] for j in train_index]
        x_train_func = [xs[j] for j in train_index]
        y_train_func = [ys[j] for j in train_index]

        for j in train_index:
            negs = id2neg[j] if j in id2neg else []
            for neg in negs:
                x_train_a.append(xs_a_neg[neg])
                x_train_b.append(xs_b_neg[neg])
                y_train_prov.append([0, 1])

                func_index = neg + len(xs_a_pos)
                x_train_func.append(xs[func_index])
                y_train_func.append(ys[func_index])

        x_train_a = np.array(x_train_a)
        x_train_b = np.array(x_train_b)
        x_train_func = np.array(x_train_func)
        y_train_func = np.array(y_train_func)
        y_train_prov = np.array(y_train_prov)

        print len(x_train_func)
        print len(y_train_func)

        x_test_a = [xs_a_pos[j] for j in test_index]
        x_test_b = [xs_b_pos[j] for j in test_index]
        y_test_prov = [[1, 0] for j in test_index]
        x_test_func = [xs[j] for j in test_index]
        y_test_func = [ys[j] for j in test_index]

        all_test_indices = list(test_index)

        for j in test_index:
            negs = id2neg[j] if j in id2neg else []
            for neg in negs:
                x_test_a.append(xs_a_neg[neg])
                x_test_b.append(xs_b_neg[neg])
                y_test_prov.append([0, 1])

                func_index = neg + len(xs_a_pos)
                x_test_func.append(xs[func_index])
                y_test_func.append(ys[func_index])
                all_test_indices.append(func_index)

        x_test_a = np.array(x_test_a)
        x_test_b = np.array(x_test_b)
        x_test_func = np.array(x_test_func)
        y_test_func = np.array(y_test_func)
        y_test_prov = np.array(y_test_prov)

        # When filter size is 256, both are better
        NB_FILTER = 256
        print 'NB_FILTER'
        print NB_FILTER

        BATCH_SIZE = 256

        # Shared layers
        shared_cnn_1 = Convolution1D(nb_filter=NB_FILTER,
                                     filter_length=5,
                                     border_mode='valid',
                                     activation='relu')
        shared_pooling_1 = MaxPooling1D(5)
        shared_cnn_2 = Convolution1D(nb_filter=NB_FILTER,
                                     filter_length=5,
                                     border_mode='valid',
                                     activation='relu')
        shared_embedding = Embedding(len(word_index) + 1,
                                     EMBEDDING_DIM,
                                     weights=[embedding_matrix],
                                     input_length=MAX_SEQUENCE_LENGTH)
        # shared_embedding.trainable = False
        # shared_dropout = Dropout(params.dropout_rate)
        shared_pooling_2 = GlobalMaxPooling1D()
        shared_dense = Dense(NB_FILTER, activation='relu')

        # Function
        func_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
        func = shared_embedding(func_input)
        func = shared_cnn_1(func)
        # func = MaxPooling1D(5)(func)
        # func = shared_cnn_2(func)
        func = GlobalMaxPooling1D()(func)
        func = shared_dense(func)
        func_model = Dense(len(funcs_index), activation='softmax')(func)

        # Provenance
        prov_a_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
        # prov_a = Embedding(len(word_index) + 1,
        #                              EMBEDDING_DIM,
        #                              weights=[embedding_matrix],
        #                              input_length=MAX_SEQUENCE_LENGTH)(prov_a_input)
        # prov_a = Convolution1D(nb_filter=NB_FILTER,
        #                              filter_length=5,
        #                              border_mode='valid',
        #                              activation='relu')(prov_a)
        prov_a = shared_embedding(prov_a_input)
        prov_a = shared_cnn_1(prov_a)
        # prov_a = MaxPooling1D(5)(prov_a)
        # prov_a = shared_cnn_2(prov_a)
        prov_a = GlobalMaxPooling1D()(prov_a)
        prov_a = shared_dense(prov_a)
        # prov_a = Dense(NB_FILTER, activation='relu')(prov_a)

        prov_b_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
        prov_b = Embedding(len(word_index) + 1,
                                     EMBEDDING_DIM,
                                     weights=[embedding_matrix],
                                     input_length=MAX_SEQUENCE_LENGTH)(prov_b_input)
        prov_b = Convolution1D(nb_filter=NB_FILTER,
                                     filter_length=5,
                                     border_mode='valid',
                                     activation='relu')(prov_b)
        # prov_b = MaxPooling1D(5)(prov_b)
        # prov_b = Convolution1D(nb_filter=128,
        #                              filter_length=5,
        #                              border_mode='valid',
        #                              activation='relu')(prov_b)
        prov_b = GlobalMaxPooling1D()(prov_b)
        prov_b = Dense(NB_FILTER, activation='relu')(prov_b)

        # prov_b_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
        # prov_b = shared_embedding(prov_b_input)
        # prov_b = shared_cnn_1(prov_b)
        # # prov_b = MaxPooling1D(5)(prov_b)
        # # prov_b = shared_cnn_2(prov_b)
        # prov_b = GlobalMaxPooling1D()(prov_b)
        # prov_b = shared_dense(prov_b)

        # prov_c = merge([prov_a, prov_b], mode='mul')
        # prov = merge([prov_a, prov_b, prov_c], mode='concat')
        prov = merge([prov_a, prov_b], mode='mul')

        # prov = merge([prov_a, prov_b], mode='concat')
        prov_model = Dense(len(provs_index), activation='softmax')(prov)

        # Combined model
        model = Model(input=[func_input, prov_a_input, prov_b_input],
                      output=[func_model, prov_model])

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['acc'])

        print model.summary()
        # print y_train.shape

        # happy learning!
        model.fit([x_train_func, x_train_a, x_train_b],
                  [y_train_func, y_train_prov],
                  nb_epoch=30, batch_size=BATCH_SIZE)

        [y_pred_func, y_pred_prov] = model.predict(
            [x_test_func, x_test_a, x_test_b])

        y_pred_func = map(lambda x: pd.Series(x).idxmax(), y_pred_func)
        y_pred_prov = map(lambda x: pd.Series(x).idxmax(), y_pred_prov)

        # Generate classification report
        y_test_func = data.compress_y(y_test_func)
        y_test_prov = data.compress_y(y_test_prov)

        print whole_dataset[:2]
        i_zero = 0
        for i in all_test_indices:
            row = [whole_dataset[i]['context'][1],
                y_test_func[i_zero],
                y_pred_func[i_zero],]
            # row = [whole_dataset[i]['context'][1],
            #     ' '.join(whole_dataset[i]['provenance']),
            #     y_test_prov[i_zero],
            #     y_pred_prov[i_zero],]
            csv_writer.writerow(row)
            i_zero += 1

        print('y_pred_func')
        print(y_pred_func)
        print('y_test_func')
        print(y_test_func)

        y_pred_func_all += y_pred_func
        y_test_func_all += y_test_func
        y_pred_prov_all += y_pred_prov
        y_test_prov_all += y_test_prov




        continue

        # ---------- Only citation function ----------
        embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        embedding_layer = Embedding(len(word_index) + 1,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH)

        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = Convolution1D(nb_filter=NB_FILTER,
                         filter_length=5,
                         border_mode='valid',
                         activation='relu')(embedded_sequences)
        # x = MaxPooling1D(5)(x)
        # x = Conv1D(128, 5, activation='relu')(x)
        # x = MaxPooling1D(5)(x)
        # x = Flatten()(x)
        # x = Dropout(0.1)(x)
        # x = BatchNormalization()(x)
        # x = Attention()(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(NB_FILTER, activation='relu')(x)
        # x = Dropout(0.5)(x)
        # x = Bidirectional(GRU(128))(embedded_sequences)
        # x = Dropout(0.1)(x)
        preds = Dense(len(funcs_index), activation='softmax')(x)

        model = Model(sequence_input, preds)

        # model.layers[1].trainable = True
        model.compile(loss='categorical_crossentropy',
                      # optimizer='adam',
                      optimizer='rmsprop',
                      metrics=['acc'])

        y_train_copy = map(lambda x : x.tolist().index(1), y_train_func)
        weight = class_weight.compute_class_weight('balanced', np.unique(y_train_copy), y_train_copy)
        print weight
        model.fit(x_train_func, y_train_func,
                  nb_epoch=30, batch_size=BATCH_SIZE, class_weight=weight)

        print model.summary()
        y_pred_probs = model.predict(x_test_func)

        y_pred_func = map(lambda x: pd.Series(x).idxmax(), y_pred_probs)

        # Generate classification report
        y_test_func = data.compress_y(y_test_func)
        print(y_pred_func)
        print(y_test_func)

        y_pred_only_func_all += y_pred_func
        y_test_only_func_all += y_test_func

        # ---------- End of citation function ----------

        # ---------- Start of citation provenance ----------

        ax = Sequential()
        ax.add(Embedding(len(word_index) + 1,
                         EMBEDDING_DIM,
                         weights=[embedding_matrix],
                         input_length=MAX_SEQUENCE_LENGTH))
        ax.add(Convolution1D(nb_filter=NB_FILTER,
                             filter_length=5,
                             border_mode='valid',
                             activation='relu'))
        ax.add(GlobalMaxPooling1D())
        ax.add(Dense(NB_FILTER, activation='relu'))

        bx = Sequential()
        bx.add(Embedding(len(word_index) + 1,
                         EMBEDDING_DIM,
                         weights=[embedding_matrix],
                         input_length=MAX_SEQUENCE_LENGTH))
        bx.add(Convolution1D(nb_filter=NB_FILTER,
                             filter_length=5,
                             border_mode='valid',
                             activation='relu'))
        bx.add(GlobalMaxPooling1D())
        bx.add(Dense(NB_FILTER, activation='relu'))

        # embedding_layer = Embedding(len(word_index) + 1,
        #                             EMBEDDING_DIM,
        #                             weights=[embedding_matrix],
        #                             input_length=MAX_SEQUENCE_LENGTH)

        # asequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        # asequence_input = Sequential()
        # aembedded_sequences = embedding_layer(asequence_input)
        # ax = Conv1D(128, 5, activation='relu')(aembedded_sequences)
        # ax = MaxPooling1D(5)(ax)
        # ax = Conv1D(128, 5, activation='relu')(ax)
        # # ax = MaxPooling1D(5)(ax)
        # # ax = Flatten()(ax)
        # ax = GlobalMaxPooling1D()(ax)
        # ax = Dense(128, activation='relu')(ax)

        # bsequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        # bembedded_sequences = embedding_layer(bsequence_input)
        # bx = Conv1D(128, 5, activation='relu')(bembedded_sequences)
        # bx = MaxPooling1D(5)(bx)
        # bx = Conv1D(128, 5, activation='relu')(bx)
        # # bx = MaxPooling1D(5)(bx)
        # # bx = Flatten()(bx)
        # bx = GlobalMaxPooling1D()(bx)
        # bx = Dense(128, activation='relu')(bx)

        # seq_features = merge([ax, bx], mode='concat')
        # # seq_features = Dense(128, activation='relu')(seq_features)
        # preds = Dense(len(ys_index), activation='softmax')(seq_features)

        # model = Model(inputs=[asequence_input, bsequence_input], outputs=preds)
        # model.compile(loss='categorical_crossentropy',
        #               optimizer='rmsprop',
        #               metrics=['acc'])


        model = Sequential()
        model.add(Merge([ax, bx], mode='concat'))
        # model.add(Dense(len(ys_index), activation='softmax'))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])

        print model.summary()
        # print y_train.shape

        # happy learning!
        model.fit([x_train_a, x_train_b], y_train_prov,
                  nb_epoch=30, batch_size=BATCH_SIZE)

        y_pred_probs = model.predict([x_test_a, x_test_b])

        import pandas as pd

        y_pred_prov = map(lambda x: pd.Series(x).idxmax(), y_pred_probs)

        # Generate classification report
        y_test_prov = data.compress_y(y_test_prov)

        y_pred_only_prov_all += y_pred_prov
        y_test_only_prov_all += y_test_prov

        # ---------- End of citation provenance ----------

    # Aggregate the results and print report

    # results = metrics.precision_recall_fscore_support(y_test, y_pred)
    # ans = []
    # for c in range(0, 4):
    #     for res in results[:-1]:
    #         ans += res[c]

    print 'MTL_Func'
    print metrics.classification_report(y_test_func_all, y_pred_func_all, digits=4)
    print 'MTL_Prov'
    print metrics.classification_report(y_test_prov_all, y_pred_prov_all, digits=4)
    print 'Plain_Func'
    print metrics.classification_report(y_test_only_func_all, y_pred_only_func_all, digits=4)
    print 'Plain_Prov'
    print metrics.classification_report(y_test_only_prov_all, y_pred_only_prov_all, digits=4)

    # ans += [metrics.precision_score(y_test, y_pred, average='weighted'),
    #         metrics.recall_score(y_test, y_pred, average='weighted'),
    #         metrics.f1_score(y_test, y_pred, average='weighted')]

    # print ans
