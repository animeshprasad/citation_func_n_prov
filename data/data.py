"""
Common data operations.
"""
import csv
import os
import json
import data_func
import data_prov
import bootstrap_prov
import config.config as config
import config.params_func as params_func
import config.params_prov as params_prov
import config.exceptions as exceptions
import lib.parse as parse
import lib.regex as regex
import lib.utils as utils
import lib.file_ops as file_ops
import lib.citfunc as citfunc
import lib.citprov as citprov
import numpy as np
import random

from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import sklearn.metrics as metrics
import sklearn.cross_validation as cross_validation


def get_data(article):
    """
    Get data instances from a given article. Example usage: get_data('A83-1005')
    """
    # Get body text, split into sentences and extract markers
    instances = []

    if parse.is_parsed(article):
        sentences = parse.get_texts(article)
    else:
        sentences = data_prov.get_prov_texts(article)
    sentences = map(utils.remove_escapechar, sentences)
    sentences = map(utils.remove_newline, sentences)

    for i, sent in enumerate(sentences):
        context = []
        for j in range(i - config.NUM_SENTENCES_PRECEDING_CITING,
                       i + config.NUM_SENTENCES_PRECEDING_CITING + 1):
            if j <= 0:
                context.append('')
            elif j >= len(sentences) - 1:
                context.append('')
            else:
                context.append(sentences[j])

        markers = utils.get_markers(sent)
        for m in markers:
            # One marker corresponds to one data instance
            item = {}
            item['article'] = article
            item['marker'] = m
            item['context'] = context
            item['current'] = sent
            item['label'] = ''
            instances.append(item)

    return instances


def read_json_data(filename):
    """
    Read the given JSON file.
    """
    if not filename.endswith(file_ops.JSON_EXTENSION):
        raise Exception(exceptions.FILE_NOT_JSON_EXCEPTION)

    path = os.path.join(config.DATA_DIR, filename)
    if file_ops.file_exists(path):
        with open(path, 'rb') as fp:
            content = json.load(fp)
            return content
    else:
        return []


def save_json_data(data, filename):
    """
    Given a dataset, save it to a JSON file.
    """
    if not filename.endswith(file_ops.JSON_EXTENSION):
        filename += file_ops.JSON_EXTENSION
    path = os.path.join(config.DATA_DIR, filename)

    with open(path, 'wb') as fp:
        json.dump(data, fp)


"""
The following operations are called in the classifiers to deal with vocabulary,
OOV words, etc.
"""


def process_data(xs, original, new_vocab, mode):
    """
    Processes the xs data into desired format for the models. Specifically,
    this function prepends sent_start tokens to each sequence, and replaces
    OOV words by the OOV token. The input data (xs) are all indices of words.
    
    The function also changes the indices in the original text into the new
    vocabulary.
    - Example: if 'the' is of index 3 in original vocab and 4 in new vocab, all
    appearances of 3 in xs will be changed to 4.
    - Assumption: in new_vocab, all indices are already offset by index_offset
    - original: the original mapping of words to their indices / original vocab
    - Reference:
        https://github.com/fchollet/keras/blob/master/keras/datasets/imdb.py
    """
    # Select the correct parameters
    if mode == 'func':
        nb_words = params_func.nb_words
        skip_top = params_func.skip_top
    else:
        nb_words = params_prov.nb_words
        skip_top = params_prov.skip_top

    # Reversed mapping of index -> word of the original vocabulary
    original_rev = {}
    for w in original.keys():
        original_rev[original[str(w)]] = w

    specials = set([config.VocabChar.MARKER_TOKEN.value.lower().strip(),
                    config.VocabChar.DECIMAL_TOKEN.value.lower().strip(),
                    config.VocabChar.NUMBER_TOKEN.value.lower().strip()])

    new_xs = []
    for x in xs:
        nx = []
        # Sentence start character
        nx.append(config.VocabChar.START.value)

        for w in x:
            nw = w
            # Change vocabulary
            if original_rev[w] in new_vocab:
                nw = new_vocab[original_rev[w]]
            else:
                nw = config.VocabChar.OOV.value

            # Deal with cases of OOV
            if nw >= nb_words:
                nw = config.VocabChar.OOV.value

            if (original_rev[w].lower() in specials) or (
                        nw >= skip_top) or (nw == config.VocabChar.OOV.value):
                nx.append(nw)

        new_xs.append(nx)

    return new_xs


def build_word_index():
    """
    Build the word_index using SemEval data, and the func, prov training data.
    No test data.
    """
    texts = []

    # SemEval data
    semeval = bootstrap_prov.read_all_data()
    for d in semeval:
        texts.append(str(' '.join(d['context'])))
        texts.append(str(' '.join(d['provenance'])))

    # CitFunc data
    func_files = ['processed/golden_train.func.json',
                  'processed/silver.func.json']
    func = []
    for f in func_files:
        func += read_json_data(f)
    for d in func:
        texts.append(str(' '.join(d['context'])))

    # CitProv data
    prov_files = ['processed/golden_train.prov.json']
    prov = []
    for f in prov_files:
        prov += read_json_data(f)
    for d in prov:
        texts.append(str(' '.join(d['context'])))
        texts.append(str(' '.join(d['provenance'])))

    texts = map(utils.replace_char, texts)

    # Tokenise and generate word-index mappings
    tokenizer = Tokenizer(filters=config.BASE_FILTER)
    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index

    # Increment by index_offset
    for w in word_index.keys():
        word_index[w] += config.VocabChar.INDEX_OFFSET.value

    with open(config.WORD_INDEX_FILEPATH, 'wb') as fp:
        json.dump(word_index, fp)


def get_word_index():
    """
    Retrieves the word index dictionary.
    """
    if not file_ops.file_exists(config.WORD_INDEX_FILEPATH):
        build_word_index()

    with open(config.WORD_INDEX_FILEPATH, 'rb') as fp:
        content = json.load(fp)
        return content


def get_class_weights(x_train, y_train):
    """
    Returns the class weights to make training classes have equal numbers.
    """
    y_train = map(lambda x: x.tolist().index(1), y_train)
    classes = set(list(y_train))
    weights = {}
    for c in classes:
        if config.USE_CLASS_WEIGHTS:
            weights[c] = float(len(y_train)) / float(
                len(filter(lambda y: y == c, y_train)))
        else:
            weights[c] = 1
    return weights


"""
Below are functions directly used by the models (i.e. the returned values
can and should be directly fed into the deep learning models.)
"""


def prepare_data_func(dataset):
    """
    Prepare the training and test func data. Returns arrays of indices
    representing the texts.
    """
    # Preparing the text data
    texts, labels = [], []
    for i, instance in enumerate(dataset):
        label = citfunc.func2int(instance['label'])
        if config.USE_ONLY_CITING_SENTENCE:
            text = utils.replace_char(str(instance['context'][1]))
        else:
            text = utils.replace_char(str(' '.join(instance['context'])))

        texts.append(text)
        labels.append(label)

    tokenizer = Tokenizer(nb_words=params_func.nb_words,
                          filters=config.BASE_FILTER)
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    sequences = process_data(sequences, tokenizer.word_index, get_word_index(),
                             mode='func')

    print('Found %s unique tokens.' % len(tokenizer.word_index))

    data = pad_sequences(sequences, maxlen=params_func.maxlen)
    labels = np_utils.to_categorical(np.asarray(labels))

    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    return (data, labels)


def prepare_data_prov_1(dataset):
    """
    Prepare the data for provenance. First architecture: citing sentence and
    provenance concatenated together.
    """
    texts, labels = [], []
    for i, instance in enumerate(dataset):
        text = '%s %s' % (str(' '.join(instance['context'])),
                          str(' '.join(instance['provenance'])))
        text = utils.replace_char(text)
        texts.append(text)

        label = citprov.prov2int(instance['label'])
        labels.append(label)

    tokenizer = Tokenizer(nb_words=params_prov.nb_words,
                          filters=config.BASE_FILTER)
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    sequences = process_data(sequences, tokenizer.word_index, get_word_index(),
                             mode='prov')

    print('Found %s unique tokens.' % len(tokenizer.word_index))

    data = pad_sequences(sequences, maxlen=params_prov.maxlen)
    labels = np_utils.to_categorical(np.asarray(labels))

    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    return (data, labels)


def prepare_data_prov_2(dataset):
    """
    Prepare the data for provenance. Second architecture: citing sentence and
    provenance using one CNN each.
    """
    cit_texts, prov_texts = [], []
    labels = []

    for i, instance in enumerate(dataset):
        cit = utils.replace_char(str(' '.join(instance['context'])))
        cit_texts.append(cit)

        prov = utils.replace_char(str(' '.join(instance['provenance'])))
        prov_texts.append(prov)

        label = citprov.prov2int(instance['label'])
        labels.append(label)

    tokenizer = Tokenizer(nb_words=params_prov.nb_words,
                          filters=config.BASE_FILTER)
    tokenizer.fit_on_texts(cit_texts + prov_texts)

    word_index = get_word_index()

    cit_seqs = tokenizer.texts_to_sequences(cit_texts)
    cit_seqs = process_data(cit_seqs, tokenizer.word_index, word_index,
                            mode='prov')

    prov_seqs = tokenizer.texts_to_sequences(prov_texts)
    prov_seqs = process_data(prov_seqs, tokenizer.word_index, word_index,
                             mode='prov')

    print('Found %s unique tokens.' % len(tokenizer.word_index))

    data_cit = pad_sequences(cit_seqs, maxlen=params_prov.maxlen)
    data_prov = pad_sequences(prov_seqs, maxlen=params_prov.maxlen)
    labels = np_utils.to_categorical(np.asarray(labels))

    print('Shape of data_cit tensor:', data_cit.shape)
    print('Shape of data_prov tensor:', data_prov.shape)
    print('Shape of label tensor:', labels.shape)

    return (data_cit, data_prov, labels)


def save_preds(ys, task, model, mode='test'):
    """
    Save the predictions along with the parameters.
    - Task: either func or prov.
    - Mode: either test or pred.
    """
    ys = '\n'.join(map(lambda y: str(y), ys))

    if mode == 'test':
        # For gold standard, save only once
        filepath = config.MODEL_RES_FILEPATH % (
            task, utils.get_last_timestamp(), 'gold', '', '0')
        if not file_ops.file_exists(filepath):
            with open(filepath, 'wb') as f:
                f.write(ys)

    elif mode == 'pred':
        param_names = ['maxlen', 'dropout_rate', 'nb_words', 'skip_top',
                       'nb_epoch', 'trainable', 'embedding_dims', 'batch_size']

        # Current parameters
        param_currs = []
        for p in param_names:
            if task == 'func':
                param_currs.append(str(params_func.__dict__[p]))
            else:
                param_currs.append(str(params_prov.__dict__[p]))
        param_currs = ','.join(param_currs)

        serial = 0
        filepath = config.MODEL_RES_FILEPATH % (
            task, utils.get_last_timestamp(), model, param_currs, str(serial))

        while file_ops.file_exists(filepath):
            serial += 1
            filepath = config.MODEL_RES_FILEPATH % (
                task, utils.get_last_timestamp(), model, param_currs,
                str(serial))

        with open(filepath, 'wb') as f:
            f.write(ys)


"""
Custom cross validation.
"""


def compress_y(ys):
    """
    For each y in ys, if y is of the form [0 0 ... 1 ... 0], compress it to a
    single integer.
    """
    if len(ys) < 1:
        return ys

    if isinstance(ys[0], np.ndarray):
        # A hack >.<
        return map(lambda x: x.tolist().index(1), ys)
    else:
        return ys


def save_cv_results(cv_indices, y_pred, y_true, mode='func'):
    """
    Save the cross validation results to a CSV file.
    - TODO Refactor this, it is a hack..
    """
    golden_train_file = 'processed/golden_train.%s.json' % (mode)
    golden_test_file = 'processed/golden_test.%s.json' % (mode)

    golden_train = read_json_data(golden_train_file)
    golden_test = read_json_data(golden_test_file)
    golden = golden_train + golden_test

    if mode == 'func':
        y_pred = map(citfunc.int2func, y_pred)
        y_true = map(citfunc.int2func, y_true)

        results_path = config.CV_RES_FILEPATH
        if not file_ops.file_exists(results_path):
            row = ['Model', 'Citing Sentence', 'Citation Context', 'Sentiment',
                   'Feature Vector', 'Label = Actual', 'Actual', 'Predicted']
            with open(results_path, 'wb') as f:
                writer = csv.writer(f)
                writer.writerow(row)

        with open(results_path, 'ab') as f:
            writer = csv.writer(f)
            counter = 0
            for i in cv_indices:
                row = ['', golden[i]['context'][1], ' '.join(golden[i]['context']),
                       golden[i]['sentiment'][0], golden[i]['vector_2'],
                       str(golden[i]['label']), y_true[counter], y_pred[counter]]
                writer.writerow(row)
                counter += 1

            writer.writerow([])
            writer.writerow([])
            writer.writerow([])
    else:
        y_pred = map(citprov.int2prov, y_pred)
        y_true = map(citprov.int2prov, y_true)

        results_path = config.CV_RES_FILEPATH
        if not file_ops.file_exists(results_path):
            row = ['Model', 'Citing Sentence', 'Target Fragment', 'Label = Actual', 'Actual', 'Predicted']
            with open(results_path, 'wb') as f:
                writer = csv.writer(f)
                writer.writerow(row)

        with open(results_path, 'ab') as f:
            writer = csv.writer(f)
            counter = 0
            for i in cv_indices:
                row = ['', ' '.join(golden[i]['context']), ' '.join(golden[i]['provenance']),
                       str(golden[i]['label']), y_true[counter], y_pred[counter]]
                writer.writerow(row)
                counter += 1

            writer.writerow([])
            writer.writerow([])
            writer.writerow([])


def cross_validate(clf, xs, ys, cv=5, features=[[], []], mode='func', silver_xs=[], silver_ys=[], whole_dataset=None):
    """
    Perform cross validation for the given classifier. Aggregate the results
    and print out the classification report. Returns all predictions in the
    order of the data given.
    """
    xs = np.array(xs)
    ys = np.array(ys)
    # silver_xs = np.array(silver_xs)
    # silver_ys = np.array(silver_ys)
    features = np.array(np.concatenate(features))

    have_features = (len(features) > 0)
    params = params_func if mode == 'func' else params_prov
    print mode

    # TODO Not sure what's the difference between the following two.
    skf = cross_validation.StratifiedKFold(compress_y(ys), n_folds=cv,
                                           shuffle=True,
                                           random_state=0)
    # skf = cv_split(xs)

    cm = None
    y_pred_all = None
    y_true_all = None
    indices = None
    print 'JLJKSAD'

    try:
        # Store initial weights
        weights = clf.get_weights()
    except Exception, e:
        weights = clf.get_params()

    import csv
    with open('/Users/suxuan/Development/prov_baseline.csv', 'wb') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Citing Sentence', 'Target', 'Actual', 'Baseline'])
        # csv_writer.writerow(['Citing Sentence', 'Actual', 'Baseline'])

        for train_index, test_index in skf:
            print train_index
            x_train, x_test = xs[train_index], xs[test_index]
            y_train, y_test = ys[train_index], ys[test_index]
            print x_train[:2]

            # x_train = np.concatenate([x_train, silver_xs])
            # y_train = np.concatenate([y_train, silver_ys])

            if not have_features:
                try:
                    # Train the classifier from scratch
                    clf.set_weights(weights)
                    clf.fit(x_train, y_train,
                            nb_epoch=50,
                            # nb_epoch=params.nb_epoch,
                            batch_size=params.batch_size,
                            class_weight=get_class_weights(x_train, y_train))

                    y_test = compress_y(y_test)
                    y_pred = clf.predict(x_test)
                    y_pred = y_pred.argmax(axis=-1)
                except Exception, e:
                    print 'Exception (ignore this): %s' % (e)
                    clf.set_params(**weights)
                    clf.fit(x_train, y_train)
                    y_pred = clf.predict(x_test)
            else:
                features_train = features[train_index]
                features_test = features[test_index]

                clf.set_weights(weights)
                clf.fit([x_train, features_train], y_train,
                        nb_epoch=params.nb_epoch,
                        batch_size=params.batch_size,
                        class_weight=get_class_weights(x_train, y_train))

                y_test = compress_y(y_test)
                y_pred = clf.predict([x_test, features_test])
                y_pred = np_utils.probas_to_classes(y_pred)

            i_from_zero = 0
            for i in test_index:
                print y_test[i_from_zero]
                print '1'
                print len(whole_dataset)
                print i
                print '2'
                print len(y_test)
                print i_from_zero
                print '3'
                print len(y_pred)
                row = [whole_dataset[i]['context'][1],
                    ' '.join(whole_dataset[i]['provenance']),
                    y_test[i_from_zero],
                    y_pred[i_from_zero]]
                # row = [whole_dataset[i]['current'],
                #     y_test[i_from_zero],
                #     y_pred[i_from_zero]]
                print (row)
                i_from_zero += 1
                csv_writer.writerow(row)

            # Collect y_pred per-fold
            if y_pred_all is None:
                y_pred_all = y_pred
            else:
                y_pred_all = np.concatenate([y_pred_all, y_pred])

            if y_true_all is None:
                y_true_all = y_test
            else:
                y_true_all = np.concatenate([y_true_all, y_test])

            if indices is None:
                indices = test_index
            else:
                indices = np.concatenate([indices, test_index])

            # Sum the cv per fold
            cv_cm = metrics.confusion_matrix(y_test, y_pred)
            if cm is None:
                cm = cv_cm
            else:
                cm += cv_cm

    # save_cv_results(indices, y_pred_all, y_true_all, mode=mode)

    # This is the correct classification report
    print metrics.classification_report(y_true_all, y_pred_all, digits=4)
    print cm
    return y_pred_all


def cross_validate_2(clf, xs_cit, xs_prov, ys, cv=5, features=[[], []],
                     mode='prov'):
    """
    Cross validation for provenance architecture 2 only.
    """
    xs_cit = np.array(xs_cit)
    xs_prov = np.array(xs_prov)
    ys = np.array(ys)
    features = np.array(np.concatenate(features))

    have_features = (len(features) > 0)
    params = params_func if mode == 'func' else params_prov

    skf = cross_validation.StratifiedKFold(compress_y(ys), n_folds=cv,
                                           shuffle=True,
                                           random_state=0)
    # skf = cv_split(xs)

    cm = None
    y_pred_all = None
    y_true_all = None
    indices = None

    # Store initial weights
    weights = clf.get_weights()

    for train_index, test_index in skf:
        x_train_cit, x_test_cit = xs_cit[train_index], xs_cit[test_index]
        x_train_prov, x_test_prov = xs_prov[train_index], xs_prov[test_index]
        y_train, y_test = ys[train_index], ys[test_index]

        if not have_features:
            # Train the classifier from scratch
            clf.set_weights(weights)
            clf.fit([x_train_cit, x_train_prov], y_train,
                    nb_epoch=params_prov.nb_epoch,
                    batch_size=params_prov.batch_size)

            y_test = compress_y(y_test)
            y_pred = clf.predict([x_test_cit, x_test_prov])
            y_pred = np_utils.probas_to_classes(y_pred)
        else:
            # TODO DO THE FOLLOWING PART FOR HAND_PICKED FEATURES
            features_train = features[train_index]
            features_test = features[test_index]

            clf.set_weights(weights)
            clf.fit([x_train_cit, x_train_prov, features_train], y_train,
                    nb_epoch=params.nb_epoch,
                    batch_size=params.batch_size,)
                    # class_weight=get_class_weights(x_train, y_train))

            y_test = compress_y(y_test)
            y_pred = clf.predict([x_test_cit, x_test_prov, features_test])
            y_pred = np_utils.probas_to_classes(y_pred)

        # Collect y_pred per-fold
        if y_pred_all is None:
            y_pred_all = y_pred
        else:
            y_pred_all = np.concatenate([y_pred_all, y_pred])

        if y_true_all is None:
            y_true_all = y_test
        else:
            y_true_all = np.concatenate([y_true_all, y_test])

        if indices is None:
            indices = test_index
        else:
            indices = np.concatenate([indices, test_index])

        # Sum the cv per fold
        cv_cm = metrics.confusion_matrix(y_test, y_pred)
        if cm is None:
            cm = cv_cm
        else:
            cm += cv_cm

    save_cv_results(indices, y_pred_all, y_true_all, mode=mode)

    # This is the correct classification report
    print metrics.classification_report(y_true_all, y_pred_all, digits=4)
    print cm
    return y_pred_all


def cv_split(dataset):
    """
    Dataset is the mtl dataset. 
    """
    # This is a hack
    if 'label' not in dataset[0]:
        dataset = data_prov.get_common_dataset()

    context2index = {}
    prov2nonprov = {}
    for index, instance in enumerate(dataset):
        context = ' '.join(instance['context'])
        if instance['label'] == 'Prov':
            context2index[context] = index
            prov2nonprov[index] = list()
        else:
            if context in context2index:
                prov_index = context2index[context]
                prov2nonprov[prov_index].append(index)

    # All provenance with corresponding non-prov
    prov_1 = filter(lambda x: len(prov2nonprov[x]) > 0, prov2nonprov.keys())
    prov_2 = filter(lambda x: len(prov2nonprov[x]) == 0, prov2nonprov.keys())
    random.seed(97)
    random.shuffle(prov_1)
    random.shuffle(prov_2)
    len_1 = len(prov_1) / params_prov.cv_fold
    len_2 = len(prov_2) / params_prov.cv_fold

    skf = []
    for i in range(0, params_prov.cv_fold):
        # Firstly, get test_index
        temp = prov_1[len_1 * i: len_1 * (i + 1)]
        test_index = []
        for index in temp:
            test_index.append(index)
            test_index += prov2nonprov[index]

        temp = prov_2[len_2 * i: len_2 * (i + 1)]
        test_index += temp

        # Then, get train_index
        train_index = range(0, len(dataset))
        train_index = filter(lambda x: x not in test_index, train_index)

        skf.append((train_index, test_index))

    return skf
