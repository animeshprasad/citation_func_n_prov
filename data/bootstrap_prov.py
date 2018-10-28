"""
This file implements data bootstrapping for citation provenance. It trains the
SemEval 2016 dataset on the prov models and saves the parameters to files.
"""
import os
import numpy as np
import config.config as config
import lib.utils as utils
import lib.citprov as citprov
import lib.citprov_clf_modes as modes

np.random.seed(103)

categories = {
    'train': ['MSRpar', 'MSRvid', 'SMTeuroparl'],
    'test-gold': ['MSRpar', 'MSRvid', 'SMTeuroparl', 'surprise.OnWN',
                  'surprise.SMTnews'],
    'sts-en-test-gs-2014': ['deft-forum', 'deft-news', 'headlines', 'images',
                            'OnWN', 'tweet-news'],
    'sts2015-en-post/data/gs': ['answers-forums', 'answers-students', 'belief',
                                'headlines', 'images']
}


def read_data(folder, category):
    path = os.path.join(config.SEMEVAL_DIR, folder)
    text_name = 'STS.input.%s.txt' % (category)
    text_path = os.path.join(path, text_name)
    label_name = 'STS.gs.%s.txt' % (category)
    label_path = os.path.join(path, label_name)

    texts = []
    labels = []

    with open(text_path, 'rb') as text_file:
        lines = text_file.readlines()
        for l in lines:
            l = l.strip()
            if len(l) > 0:
                l = l.split('\t')
                texts.append((l[0], l[1]))

    with open(label_path, 'rb') as label_file:
        lines = label_file.readlines()
        for i, l in enumerate(lines):
            l = l.strip()
            labels.append(l)

    assert len(texts) == len(labels)
    return (texts, labels)


def read_all_data():
    """
    Read data from all categories.
    """
    # Initialize dataset
    instances = []
    for folder in categories.keys():
        for category in categories[folder]:
            dataset = read_data(folder, category)
            texts = dataset[0]  # Each entry: cit -> prov pair
            labels = dataset[1]

            for i in range(0, len(texts)):
                if len(labels[i]) == 0:
                    continue

                item = {}
                item['cited'] = 'SemEval2016'
                item['citing'] = 'SemEval2016'
                item['marker'] = ''
                item['context'] = [utils.remove_special(texts[i][0])]
                item['provenance'] = [utils.remove_special(texts[i][1])]
                item['label'] = 'Prov' if float(labels[i]) >= 3 else 'Non-Prov'
                instances.append(item)
    return instances


def init(models, ratio=1):
    """
    Generate the models and save it.
    """
    instances = read_all_data()

    # Train-test split
    train_indices, test_indices = utils.get_mlsets_indices(len(instances))
    training = map(instances.__getitem__, train_indices)
    test = map(instances.__getitem__, test_indices)

    assert (ratio <= 1)
    training = utils.get_random(training, int(len(training) * ratio))
    test = utils.get_random(test, int(len(test) * ratio))

    for model in models:
        model.classify(training, test, mode=modes.MODE_SEMEVAL_BTSP)
