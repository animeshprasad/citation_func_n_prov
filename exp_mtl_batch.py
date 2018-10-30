# New file authored 28 Jan 2018
# For citation provenance
# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

"""
Perform the experiments on bootstrapped data and actual annotated data.
"""
import lib.logger, os, sys, random, math
import numpy as np

import config.config as config
import data.data as data
import data.data_func as data_func
import sklearn.metrics as metrics
from sklearn.model_selection import KFold
import pandas as pd
from tensorflow.python import debug as tf_debug

from sklearn.utils import class_weight

import keras.backend as K
from keras.utils import np_utils
from keras.engine.topology import Input
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, merge, \
    GlobalMaxPooling1D, Merge, Embedding, Dropout, Masking
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

print len(dataset_pos), len(dataset_neg)

# Provenance dataset end

# Function dataset start
datafiles = config.DATA_FILES['func']
golden_train = data.read_json_data(datafiles['golden_train'])
golden_test = data.read_json_data(datafiles['golden_test'])

dataset_func = filter(lambda d: d['label'] != 'Error',
                      golden_train + golden_test)


print (len(dataset), len(dataset_func))

#lendiff = len(dataset) - len(dataset_func)
#print lendiff
#dataset_func += random.sample(dataset_func, lendiff)
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

#whole_dataset = dataset_func
#whole_dataset = dataset_pos + dataset_neg

import csv
print (id2neg)
with open('outfile.csv', 'wb') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['Citing', 'Actual', 'MTL'])
    # csv_writer.writerow(['Citing', 'Target', 'Actual', 'MTL'])

    # split the data into a training set and a validation set
    for func_split, prov_split in zip(kf.split(xs), kf.split(xs_a_pos)):
        x_train_func = [xs[j] for j in func_split[0]]
        y_train_func = [ys[j] for j in func_split[0]]
        x_train_a = [xs_a_pos[j] for j in prov_split[0]]
        x_train_b = [xs_b_pos[j] for j in prov_split[0]]
        y_train_prov = [[1, 0] for j in prov_split[0]]

        print (len(prov_split[0]))
        print ('----->')

        ka = 0
        pa =0
        for j in prov_split[0]:
            negs = id2neg[j] if j in id2neg else []
            if negs == []:
                ka += 1
            else:
                pa += 1
            for neg in negs:
                x_train_a.append(xs_a_neg[neg])
                #print (xs_a_neg[id2neg[ka[0]]][0])
                x_train_b.append(xs_b_neg[neg])
                #print (xs_b_neg[id2neg[ka[0]]][0])
                y_train_prov.append([0, 1])


        print (ka, pa)


        x_train_a = np.array(x_train_a)
        x_train_b = np.array(x_train_b)
        y_train_prov = np.array(y_train_prov)
        x_train_func = np.array(x_train_func)
        y_train_func = np.array(y_train_func)


        #print len(x_train_func)
        #print len(y_train_func)

        x_test_func = [xs[j] for j in func_split[1]]
        y_test_func = [ys[j] for j in func_split[1]]
        x_test_a = [xs_a_pos[j] for j in prov_split[1]]
        x_test_b = [xs_b_pos[j] for j in prov_split[1]]
        y_test_prov = [[1, 0] for j in prov_split[1]]

        print (len(prov_split[1]))
        print ('----->')

        ka = 0
        pa =0
        for j in prov_split[1]:
            negs = id2neg[j] if j in id2neg else []
            if negs == []:
                ka += 1
            else:
                pa += 1
            for neg in negs:
                x_test_a.append(xs_a_pos[j])
                x_test_b.append(xs_b_neg[neg])
                y_test_prov.append([0, 1])

        print (ka, pa)



        x_test_a = np.array(x_test_a)
        x_test_b = np.array(x_test_b)
        x_test_func = np.array(x_test_func)
        y_test_func = np.array(y_test_func)
        y_test_prov = np.array(y_test_prov)


        print (x_test_a.shape, x_test_b.shape, x_test_func.shape, y_test_func.shape, y_test_prov.shape)
        print (x_train_a.shape, x_train_b.shape, x_train_func.shape, y_train_func.shape, y_train_prov.shape)

        # sess = K.get_session()
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # K.set_session(sess)


        # When filter size is 256, both are better
        NB_FILTER = 128
        print 'NB_FILTER'
        print NB_FILTER

        BATCH_SIZE = 32

        embedding_layer1 = Embedding(len(word_index) + 1,
                                     EMBEDDING_DIM,
                                     weights=[embedding_matrix],
                                     input_length=MAX_SEQUENCE_LENGTH)

        sequence_input1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        mask1 = Masking(mask_value=0)(sequence_input1)
        embedded_sequences1 = embedding_layer1(mask1)
        x1 = Conv1D(128, 5, activation='relu')(embedded_sequences1)

        xA = GlobalMaxPooling1D()(x1)
        x1 = Dropout(Dense(32, activation='relu'))(xA)

        embedding_layer2 = Embedding(len(word_index) + 1,
                                     EMBEDDING_DIM,
                                     weights=[embedding_matrix],
                                     input_length=MAX_SEQUENCE_LENGTH)

        sequence_input2 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        mask2 = Masking(mask_value=0)(sequence_input2)
        embedded_sequences2 = embedding_layer2(mask2)
        x2 = Conv1D(128, 5, activation='relu')(embedded_sequences2)

        x2 = GlobalMaxPooling1D()(x2)
        x2 = Dropout(Dense(32, activation='relu'))(x2)

        x = Merge(mode='mul')([x1, x2])
        x = Merge(mode='concat')([x1, x2, x])
        preds_p = Dense(2, activation='softmax')(x)

        model2 = Model([sequence_input1, sequence_input2], preds_p)



        x = Dense(NB_FILTER, activation='relu')(xA)
        # # x = Dropout(0.5)(x)
        # # x = Bidirectional(GRU(128))(embedded_sequences)
        # # x = Dropout(0.1)(x)
        preds_f = Dense(len(funcs_index), activation='softmax')(x)

        model1 = Model(sequence_input1, preds_f)


        model2.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])


        model1.compile(loss='categorical_crossentropy',
                       # optimizer='adam',
                       optimizer='rmsprop',
                       metrics=['acc'])




        # # Shared layers
        # shared_cnn_1 = Convolution1D(nb_filter=NB_FILTER,
        #                              filter_length=5,
        #                              border_mode='valid',
        #                              activation='relu')
        # shared_cnn_2 = Convolution1D(nb_filter=NB_FILTER,
        #                              filter_length=5,
        #                              border_mode='valid',
        #                              activation='relu')
        # shared_embedding = Embedding(len(word_index) + 1,
        #                              EMBEDDING_DIM,
        #                              weights=[embedding_matrix],
        #                              input_length=MAX_SEQUENCE_LENGTH, trainable=False)
        # #shared_embedding.trainable = False
        # # shared_dropout = Dropout(params.dropout_rate)
        # shared_pooling_2 = GlobalMaxPooling1D()
        # shared_dense = Dropout(Dense(NB_FILTER/2, activation='relu'))
        #
        # # Function
        # func_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
        # func = shared_embedding(func_input)
        # func = shared_cnn_1(func)
        # # func = MaxPooling1D(5)(func)
        # # func = shared_cnn_2(func)
        # func = shared_pooling_2(func)
        # func1 = shared_dense(func)
        # func_model = Dense(len(funcs_index), activation='softmax')(func1)
        #
        #
        #
        # #prov_a = Dropout(Dense(NB_FILTER/2, activation='relu'))(func)
        #
        # prov_b_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
        # prov_b = Embedding(len(word_index) + 1,
        #                              EMBEDDING_DIM,
        #                              weights=[embedding_matrix],
        #                              input_length=MAX_SEQUENCE_LENGTH, trainable=False)(prov_b_input)
        #
        # prov_b = Convolution1D(nb_filter=NB_FILTER,
        #                              filter_length=5,
        #                              border_mode='valid',
        #                              activation='relu')(prov_b)
        # # prov_b = MaxPooling1D(5)(prov_b)
        # # prov_b = Convolution1D(nb_filter=128,
        # #                              filter_length=5,
        # #                              border_mode='valid',
        # #                              activation='relu')(prov_b)
        # prov_b = GlobalMaxPooling1D()(prov_b)
        # prov_b = Dropout(Dense(NB_FILTER/2, activation='relu'))(prov_b)
        #
        # # prov_b_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
        # # prov_b = shared_embedding(prov_b_input)
        # # prov_b = shared_cnn_1(prov_b)
        # # # prov_b = MaxPooling1D(5)(prov_b)
        # # # prov_b = shared_cnn_2(prov_b)
        # # prov_b = GlobalMaxPooling1D()(prov_b)
        # # prov_b = shared_dense(prov_b)
        #
        # # prov_c = merge([prov_a, prov_b], mode='mul')
        # # prov = merge([prov_a, prov_b, prov_c], mode='concat')
        # prov1 = merge([func1, prov_b], mode='mul')
        #
        # prov = merge([func1, prov_b, prov1], mode='concat')
        # prov_model = Dense(len(provs_index), activation='softmax')(prov)

        # Combined model
        # model1 = Model(input=func_input,
        #               output=func_model)
        # model2 = Model(input=[func_input, prov_b_input],
        #               output=prov_model)
        #
        # model1.compile(optimizer='rmsprop',
        #               loss='categorical_crossentropy',
        #               metrics=['acc'])
        #
        # model2.compile(optimizer='rmsprop',
        #               loss='categorical_crossentropy',
        #               metrics=['acc'])

        print model1.summary()
        print model2.summary()
        # print y_train.shape


        count = 0
        EPOCH = 30
        indices= []
        indices_type = []
        for i in range((x_train_func.shape[0])/BATCH_SIZE):
            indices.append((i*BATCH_SIZE, min((i+1)*BATCH_SIZE, x_train_func.shape[0])))
            indices_type.append(0)
        for i in range((x_train_a.shape[0]) / BATCH_SIZE):
            indices.append((i * BATCH_SIZE, min((i + 1) * BATCH_SIZE, x_train_a.shape[0])))
            indices_type.append(1)

        print (indices)


        y_train_copy = map(lambda x: x.tolist().index(1), y_train_func)
        weight_f = class_weight.compute_class_weight('balanced', np.unique(y_train_copy), y_train_copy)
        print weight_f

        y_train_copy = map(lambda x: x.tolist().index(1), y_train_prov)
        weight_p = class_weight.compute_class_weight('balanced', np.unique(y_train_copy), y_train_copy)
        print weight_p
        for i in [0, 1]:
            weight_p[i] = weight_p[i]*weight_p[i]



        while count < EPOCH:

            model2.fit([x_train_a, x_train_b], y_train_prov,
                  nb_epoch=1, batch_size=BATCH_SIZE, class_weight=weight_p)
            model1.fit(x_train_func, y_train_func,
                   nb_epoch=1, batch_size=BATCH_SIZE, class_weight=weight_f)

            count += 1


        # # happy learning!
        #     loss_func_acc = 0
        #     loss_func = 0
        #     loss_prov_acc= 0
        #     loss_prov = 0
        #     for i in  range(len(indices)):
        #         #if indices_type[i] == 0:
        #         #    x_train_funcb = x_train_func[indices[i][0]: indices[i][1]]
        #         #    y_train_funcb = y_train_func[indices[i][0]: indices[i][1]]
        #         #    #print ('Training func')
        #         #    loss1 = model1.train_on_batch(x_train_funcb,
        #         #        y_train_funcb, class_weight=weight_f)
        #         #    #print (loss1)
        #         #    loss_func += loss1[0]*x_train_funcb.shape[0]
        #         #    loss_func_acc += loss1[1]*x_train_funcb.shape[0]
        #
        #
        #         if indices_type[i] == 1:
        #             x_train_ab = x_train_a[indices[i][0]: indices[i][1]]
        #
        #             x_train_bb = x_train_b[indices[i][0]: indices[i][1]]
        #             y_train_provb = y_train_prov[indices[i][0]: indices[i][1]]
        #             #print ('Training prov')
        #             loss2 = model2.train_on_batch([x_train_ab, x_train_bb],
        #                y_train_provb,  class_weight=weight_p)
        #             #print (loss2)
        #             loss_prov += loss2[0]*x_train_ab.shape[0]
        #             loss_prov_acc += loss2[1]*x_train_ab.shape[0]

            print ( map(lambda x: pd.Series(x).idxmax(),model2.predict([x_test_a, x_test_b])))
         #   count = count + 1

            #print ('Func: loss is ', loss_func/x_train_func.shape[0], ', acc is ', loss_func_acc /x_train_func.shape[0] )
            #print ('Prov: loss is ',loss_prov/x_train_a.shape[0], ', acc is ', loss_prov_acc /x_train_a.shape[0])



        y_pred_func = model1.predict(
            x_test_func)

        y_pred_prov = model2.predict(
            [x_test_a, x_test_b])



        y_pred_func = map(lambda x: pd.Series(x).idxmax(), y_pred_func)
        y_pred_prov = map(lambda x: pd.Series(x).idxmax(), y_pred_prov)

        # Generate classification report
        y_test_func = data.compress_y(y_test_func)
        y_test_prov = data.compress_y(y_test_prov)

        print 'MTL_Func'
        print metrics.classification_report(y_test_func, y_pred_func, digits=4)
        print 'MTL_Prov'
        print metrics.classification_report(y_test_prov, y_pred_prov, digits=4)



        # print whole_dataset[:2]
        # i_zero = 0
        # for i in all_test_indices:
        #     row = [whole_dataset[i]['context'][1],
        #         y_test_func[i_zero],
        #         y_pred_func[i_zero],]
        #     # row = [whole_dataset[i]['context'][1],
        #     #     ' '.join(whole_dataset[i]['provenance']),
        #     #     y_test_prov[i_zero],
        #     #     y_pred_prov[i_zero],]
        #     csv_writer.writerow(row)
        #     i_zero += 1

        print('y_pred_prov')
        print(y_pred_prov)
        print('y_test_prov')
        print(y_test_prov)

        #print('y_pred_func')
        #print(y_pred_func)
        print('y_test_func')
        print(y_test_func)
        #raw_input()

        #y_pred_func_all += y_pred_func
        y_test_func_all += y_test_func
        y_pred_prov_all += y_pred_prov
        y_test_prov_all += y_test_prov



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


        model.fit(x_train_func, y_train_func,
                  nb_epoch=EPOCH, batch_size=BATCH_SIZE, class_weight=weight_f)

        print model.summary()
        y_pred_probs = model.predict(x_test_func)

        y_pred_func = map(lambda x: pd.Series(x).idxmax(), y_pred_probs)

        # Generate classification report
        y_test_func = data.compress_y(y_test_func)

        print('y_pred_func_A')
        print(y_pred_func)


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

        cx = Merge([ax, bx], mode='mul')

        model = Sequential()
        model.add(Merge([ax, bx, cx], mode='concat'))
        # model.add(Dense(len(ys_index), activation='softmax'))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])

        print model.summary()
        # print y_train.shape

        # happy learning!
        count = 0
        while count < EPOCH:
            model.fit([x_train_a, x_train_b], y_train_prov,
                  nb_epoch=1, batch_size=BATCH_SIZE, class_weight=weight_p)
            count += 1



        count  = 0
        while count < EPOCH:
        # happy learning!
            loss_func_acc = 0
            loss_func = 0
            loss_prov_acc= 0
            loss_prov = 0
            for i in  range(len(indices)):
                #if indices_type[i] == 0:
                #    x_train_funcb = x_train_func[indices[i][0]: indices[i][1]]
                #    y_train_funcb = y_train_func[indices[i][0]: indices[i][1]]
                #    #print ('Training func')
                #    loss1 = model1.train_on_batch(x_train_funcb,
                #        y_train_funcb, class_weight=weight_f)
                #    #print (loss1)
                #    loss_func += loss1[0]*x_train_funcb.shape[0]
                #    loss_func_acc += loss1[1]*x_train_funcb.shape[0]


                if indices_type[i] == 1:
                    x_train_ab = x_train_a[indices[i][0]: indices[i][1]]
                    x_train_bb = x_train_b[indices[i][0]: indices[i][1]]
                    y_train_provb = y_train_prov[indices[i][0]: indices[i][1]]
                    # print ('Training prov')
                    # loss2 = model2.train_on_batch([x_train_ab, x_train_bb],
                    #    y_train_provb,  class_weight=weight_p)
                    # #print (loss2)
                    # loss_prov += loss2[0]*x_train_ab.shape[0]
                    # loss_prov_acc += loss2[1]*x_train_ab.shape[0]

            print ( map(lambda x: pd.Series(x).idxmax(), model.predict([x_test_a, x_test_b])))
            count +=1



        y_pred_probs = model.predict([x_test_a, x_test_b])



        y_pred_prov = map(lambda x: pd.Series(x).idxmax(), y_pred_probs)

        # Generate classification report
        y_test_prov = data.compress_y(y_test_prov)

        print('y_pred_prov_B')
        print(y_pred_prov)
        raw_input()

        y_pred_only_prov_all += y_pred_prov
        y_test_only_prov_all += y_test_prov

        # ---------- End of citation provenance ----------

    # Aggregate the results and print report

    # results = metrics.precision_recall_fscore_support(y_test, y_pred)
    # ans = []
    # for c in range(0, 4):
    #     for res in results[:-1]:
    #         ans += res[c]

    #print 'MTL_Func'
    #print metrics.classification_report(y_test_func_all, y_pred_func_all, digits=4)
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
