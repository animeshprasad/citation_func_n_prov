"""
This file supports operations to parse a given PDF file to get data instances.
It also supports the generation of dataset with feature vectors, although
the annotation procedure still needs manual efforts. It has methods for
preparing an initial dataset for annotation.
"""
import os
import copy
import string
import data
import lib.parse as parse
import lib.utils as utils
import lib.file_ops as file_ops
import config.config as config
import config.exceptions as exceptions
import numpy as np
from lxml import etree
from pipes import quote


def get_all_articles():
    """
    Return all articles in the ACL ARC.
    """
    dirs = []
    letters = ['A', 'D', 'E', 'J', 'K', 'N', 'P', 'Q', 'S', 'W']
    for l in letters:
        upper_dir = os.path.join(config.ACL_ARC_DIR, l)
        for d in file_ops.get_dirs_in_dir(upper_dir):
            new_dir = os.path.join(upper_dir, d)
            dirs.append(new_dir)

    articles = []
    for d in dirs:
        for f in file_ops.get_files_in_dir(d):
            articles.append(f[0:8])
    return articles


def get_articles_infile(filename):
    """
    Get the list of articles listed in a file.
    """
    filepath = os.path.join(config.DATA_DIR, filename)
    lines = []
    with open(filepath, 'rb') as fp:
        lines = fp.readlines()
        lines = map(lambda x: x.strip(), lines)
    return lines


def get_data(article):
    """
    Get data instances from a given article. Example usage: get_data('A83-1005')
    """
    # Get body text, split into sentences and extract markers
    instances = []

    sentences = parse.get_texts(article)
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


def get_dataset(index):
    """
    Return all data instances in the indexed dataset. Example usage:
    get_dataset('A'), get_dataset('A00')
    """
    dirs = []  # All directories to examine
    upper_dir = os.path.join(config.ACL_ARC_DIR, index[0])
    if len(index) == 1:
        for d in file_ops.get_dirs_in_dir(upper_dir):
            new_dir = os.path.join(upper_dir, d)
            dirs.append(new_dir)
    elif len(index) == 3:
        upper_dir = os.path.join(upper_dir, index)
        dirs.append(upper_dir)

    articles = []
    for d in dirs:
        for f in file_ops.get_files_in_dir(d):
            articles.append(f[0:8])

    instances = []
    for a in articles:
        print a
        instances += get_data(a)
    return instances


"""
Data loading and saving, in txt and json formats.
- json has been moved to data.py
"""


def save_data_for_annotation(data):
    """
    Given a dataset, save it to be fed into the annotation system (i.e. txt
    format).
    """
    content = []
    for instance in data:
        context = map(utils.sent_add_tag, instance['context'])
        s = instance['article'] + '!=' + '<context citStr=' + quote(
            instance['marker']) + '>' + ' '.join(context) + '</context>'
        content.append(s)

    path = config.ANNOTATION_FILEPATH
    with open(path, 'wb') as fp:
        fp.write('\n'.join(content))


def save_articles_for_annotation(data):
    """
    Save the list of articles used as txt format.
    """
    path = config.ANNOTATION_ARTICLES_FILEPATH
    with open(path, 'wb') as fp:
        fp.write('\n'.join(data))


def read_citfunc_data(filename):
    """
    Given a txt file located at DATA_DIR, return its data as a json object.
    """
    if not filename.endswith(file_ops.TXT_EXTENSION):
        raise Exception(exceptions.FILE_NOT_TXT_EXCEPTION)

    path = os.path.join(config.DATA_DIR, filename)
    if file_ops.file_exists(path):
        instances = []
        with open(path, 'rb') as fp:
            lines = fp.readlines()
            for l in lines:
                l = l.strip().split('!=')
                article = l[0]
                context = l[1]
                label = l[2].split(',')[1]

                context = etree.fromstring(context,
                                           parser=etree.XMLParser(recover=True))
                marker = context.attrib['citStr']
                context_len = 2 * config.NUM_SENTENCES_PRECEDING_CITING + 1
                context_arr = [''] * (context_len)

                # Calculate the index of the citing sentence in context
                citing_i = (len(context) - 1) / 2
                for i, s in enumerate(context):
                    if marker in s.text:
                        citing_i = i
                for i, s in enumerate(context):
                    new_i = i - citing_i + config.NUM_SENTENCES_PRECEDING_CITING
                    context_arr[new_i] = s.text

                item = {}
                item['article'] = article
                item['marker'] = marker
                item['context'] = context_arr
                item['current'] = context[citing_i].text
                item['label'] = label
                instances.append(item)

            return instances
    else:
        return []


"""
Function for preparing dataset for annotation.
"""


def init_dataset():
    """
    Initialize dataset for annotation. Also save it at desired location.
    """
    articles = get_all_articles()
    articles = utils.get_random(articles, int(len(articles) / 2))

    instances = []
    for i in indices:
        try:
            instances += get_data(a)
        except Exception, e:
            print a
            continue
        else:
            print a
            print len(instances)
            if len(instances) > 20000:
                break

    save_data_for_annotation(instances)
    save_articles_for_annotation(articles)


def get_common_dataset():
    """
    Return the common dataset used for normal citation function and MTL
    experiments.
    - Details: composed mainly of citprov dataset + collected minority classes
    """
    datafiles = config.DATA_FILES['mtl']
    golden_train = data.read_json_data(datafiles['golden_train'])
    golden_test = data.read_json_data(datafiles['golden_test'])
    golden = golden_train + golden_test

    # Sort them into provenance and non provenance
    func_data_prov = []
    func_data_nonprov = []

    for i in golden:
        item = copy.deepcopy(i)
        item['label'] = i['function']
        item['article'] = i['citing']
        if i['label'] == 'Prov':
            func_data_prov.append(item)
        else:
            func_data_nonprov.append(item)
    return func_data_prov + func_data_nonprov

    prov_data = golden

    # Add more data to func to make it the same length as prov
    func_files = config.DATA_FILES['func']
    golden_func_train = data.read_json_data(func_files['golden_train'])
    golden_func_test = data.read_json_data(func_files['golden_test'])
    golden_func = golden_func_train + golden_func_test
    func_data += filter(lambda x: x['label'] != 'Neut', golden_func)
    func_data += filter(lambda x: x['label'] == 'Neut', golden_func)[
                 :(len(prov_data) - len(func_data))]

    return func_data
