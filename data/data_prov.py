"""
This file supports operations to retrieve dataset for citation provenance,
including dev, test, and training datasets.
"""
import copy
import csv
import os
import random
import xml.etree.ElementTree as ET

import config.config as config
import config.exceptions as exceptions
import data
import data_func
import lib.file_ops as file_ops
import lib.parse as parse
import lib.utils as utils
from lxml import etree

id2path = {'Citance_XML': {}, 'Reference_XML': {}}  # Article ID to filepath


def init():
    global id2path
    if len(id2path['Citance_XML']) == 0 or len(id2path['Reference_XML']) == 0:
        dirs = [config.SCISUMM_DEV_DIR, config.SCISUMM_TEST_DIR,
                config.SCISUMM_TRAIN_DIR]
        folders = ['Citance_XML', 'Reference_XML']

        for directory in dirs:
            for d in file_ops.get_dirs_in_dir(directory):
                for folder in folders:
                    path = os.path.join(directory, '%s/%s' % (d, folder))

                    files = file_ops.get_files_in_dir(path)
                    files = filter(lambda x: file_ops.is_xml(x), files)

                    for f in files:
                        id2path[folder][f[0:8]] = os.path.join(path, f)


def get_prov_texts(article):
    """
    Read the texts of articles from SciSumm dataset.
    """
    init()

    if article in id2path['Citance_XML']:
        filepath = id2path['Citance_XML'][article]
    elif article in id2path['Reference_XML']:
        filepath = id2path['Reference_XML'][article]
    else:
        return []

    # TODO Is latin-1 okay?
    parsed = ET.parse(filepath,
                      parser=etree.XMLParser(recover=True,
                                             encoding='latin1')).getroot()
    texts = parsed.xpath('*/S/text()')
    texts = map(
        lambda x: str(utils.convert_escapechar(utils.remove_special(x))),
        texts)

    return texts


def get_prov_frags(article, length):
    """
    Read the texts of articles from SciSumm dataset. Return the texts as a list
    of fragments, where each fragment is of the specified length.
    """
    texts = get_prov_texts(article)
    ans = []
    for i in range(0, len(texts) - length):
        ans.append(texts[i:(i + length)])
    return ans


def read_citprov_data(filepath):
    """
    Given a txt file containing provenance data, read it and return as a json
    object.
    """
    if not filepath.endswith(file_ops.TXT_EXTENSION):
        raise Exception(exceptions.FILE_NOT_TXT_EXCEPTION)

    if file_ops.file_exists(filepath):
        instances = []
        with open(filepath, 'rb') as fp:
            lines = fp.readlines()
            for l in lines:
                l = l.strip()
                if len(l) < 1:
                    continue

                # Relevant fields of l: [1] reference article, [2] citing article,
                # [4] citation marker, [6] citation text, [8] reference text
                l = l.split(' | ')
                cited = l[1].strip()[18:].strip()[0:8]
                citing = l[2].strip()[15:].strip()[0:8]
                marker = l[4].strip()[16:].strip()

                citing_s = utils.sents_remove_tag(l[6].strip()[14:].strip())
                citing_s = str(
                    utils.convert_escapechar(utils.remove_special(citing_s)))

                provenance = utils.sents_remove_tag(l[8].strip()[15:].strip())
                provenance = str(
                    utils.convert_escapechar(utils.remove_special(provenance)))
                provenance_list = utils.get_sentences(provenance)

                # Expand the citing sentence to the context by reading off the
                # citing article manually
                context = utils.get_sentences(citing_s)
                # TODO This is very slow...
                # for citation in data.get_data(citing):
                #     marker_match = utils.process_sentence(
                #         citation['marker']) == utils.process_sentence(
                #         marker) or citation['marker'] in marker or \
                #                    marker in citation['marker']
                #     citing_match = utils.process_sentence(
                #         citation['current']) == utils.process_sentence(
                #         citing_s)
                #     if marker_match and citing_match:
                #         context = citation['context']
                #         break

                item = {}
                item['cited'] = cited
                item['citing'] = citing
                item['marker'] = marker
                item['context'] = context
                item['provenance'] = provenance_list
                item['label'] = 'Prov'
                instances.append(item)

            return instances
    else:
        return []


def read_csv_crowdflower(filename):
    """
    Provenance data is annotated with functions. Read from the given CSV file.
    """
    directory = config.DATA_DIR
    filename = 'crowdflower/%s' % (filename)
    filepath = os.path.join(directory, filename)

    instances = []
    with open(filepath, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == '_unit_id':
                # Skip first row
                continue

            provenance = row[14]
            provenance = utils.get_sentences(provenance)
            item = {'marker': row[11], 'citing': row[7],
                    'context': ['', row[8], ''], 'provenance': provenance,
                    'function': row[5], 'label': 'Prov'}
            instances.append(item)
    return instances


def get_dataset(directory):
    """
    Given a data directory (either dev, test or training), return the dataset
    content.
    """
    if file_ops.file_exists(directory):
        files = []  # All files to consider

        for d in file_ops.get_dirs_in_dir(directory):
            filename = '%s/annotation/%s.annv3.txt' % (d, d[0:8])
            filepath = os.path.join(directory, filename)
            files.append(filepath)

        instances = []
        for f in files:
            f_data = read_citprov_data(f)
            instances.extend(f_data)
        return instances
    else:
        return []


def get_prov_datasets():
    """
    Get a dataset where all instances are provenances.
    """
    original = []
    original.extend(get_dataset(config.SCISUMM_DEV_DIR))
    original.extend(get_dataset(config.SCISUMM_TEST_DIR))
    original.extend(get_dataset(config.SCISUMM_TRAIN_DIR))

    # We don't have ParsCit for these
    original = filter(lambda x: x['citing'][0] not in ('C', 'I', 'H', 'M'),
                      original)

    annotated = read_csv_crowdflower('a988939 2.csv')
    for i in range(0, len(original)):
        annotated[i]['cited'] = original[i]['cited']

    return annotated


def get_nonprov_datasets():
    """
    Get a dataset where all instances are non-provenances.
    """
    instances = []
    # articles_filter = get_dataset(config.SCISUMM_TEST_DIR)
    # articles_filter = set(map(lambda x: x['cited'], articles_filter))
    seed = get_prov_datasets()
    # seed = filter(lambda x: x['cited'] in articles_filter, seed)

    lower, higher = 0.3, 0.7
    print(lower, higher)

    for s in seed:
        cited = s['cited']
        true_prov = s['provenance']
        frags = get_prov_frags(cited, len(true_prov))
        random.shuffle(frags)

        s_instances = []
        for false_prov in frags:
            to_add = False

            # Ensure that the false prov looks similar to the true prov
            false_sent = ' '.join(false_prov)
            true_sent = ' '.join(true_prov)
            set_1 = set(false_sent.split())
            set_2 = set(true_sent.split())
            if lower <= utils.jaccard(set_1, set_2) <= higher:
                to_add = True

            for sent in false_prov:
                set_1 = set(sent.split())

                for prov_sent in true_prov:
                    set_2 = set(prov_sent.split())
                    if utils.jaccard(set_1, set_2) >= 0.9:
                        to_add = False
                        break

            if to_add:
                item = {}
                item['cited'] = cited
                item['citing'] = s['citing']
                item['marker'] = s['marker']
                item['context'] = s['context']
                item['provenance'] = false_prov
                item['provenance_real'] = true_prov
                item['label'] = 'Non-Prov'
                item['function'] = s['function']
                s_instances.append(item)

        instances.extend(s_instances)

    # We don't have ParsCit for these
    instances = filter(lambda x: x['citing'][0] not in ('C', 'I', 'H', 'M'),
                       instances)
    return instances


def get_datasets():
    results = get_prov_datasets() + get_nonprov_datasets()
    return results


def get_common_dataset():
    """
    Get the common dataset for multitask learning.
    """
    datafiles = config.DATA_FILES['mtl']
    golden_train = data.read_json_data(datafiles['golden_train'])
    golden_test = data.read_json_data(datafiles['golden_test'])
    golden = golden_train + golden_test

    # Put Prov first before Non-Prov
    prov = filter(lambda x: x['label'] == 'Prov', golden)
    nonprov = filter(lambda x: x['label'] == 'Non-Prov', golden)
    return prov + nonprov
