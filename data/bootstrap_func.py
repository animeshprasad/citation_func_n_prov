"""
This file implements data bootstrapping, for generating silver datasets used
for training; for citation functions specifically.
"""
import data_func
import google
import sys
import lib.parse as parse
import lib.utils as utils
import config.config as config
import numpy as np

# Number of articles used for bootstrapping
num_articles = 5

annotate = set(
    data_func.get_articles_infile(config.ANNOTATION_ARTICLES_FILEPATH))
mturk = set(data_func.get_articles_infile(
    config.ANNOTATION_MTURK_ARTICLES_FILEPATH_1) + data_func.get_articles_infile(
    config.ANNOTATION_MTURK_ARTICLES_FILEPATH_2))

articles = data_func.get_all_articles()
articles = filter(lambda x: x not in annotate and x not in mturk, articles)
articles = utils.get_random(articles, num_articles)

# Use these articles for bootstrapping
booted_data = []

for a in articles:
    print a
    dataset = data.get_data(a)
    citing_sentences = set()

    # Idea 1: Classify citing sentences with sentiment score <= -0.6 as Weak,
    # >= 0.6 as Pos
    for d in dataset:
        score = google.get_sentiment(d['current']).sentiment.score
        if abs(score) >= 0.6:
            print d['current']
            if score < 0:
                d['label'] = 'Weak'
            else:
                d['label'] = 'Pos'
            citing_sentences.add(d['current'])
            booted_data.append(d)

    # Idea 2: Use cue words to classify the citations into particular classes
    keywords = common.func_bootstrapping_keywords

    # When these keywords exist together in citing sentence, classify as certain
    # label
    coexist_keywords = {
        # 'Weak': [['while', 'only']]
    }

    for label in keywords.keys():
        words = keywords[label]
        for w in words:
            w = w.lower()
            for d in dataset:
                if d['current'] not in citing_sentences and w in d[
                    'current'].lower():
                    d['label'] = label
                    citing_sentences.add(d['current'])
                    booted_data.append(d)

    for label in coexist_keywords.keys():
        multiwords = coexist_keywords[label]
        for mw in multiwords:
            for d in dataset:
                if d['current'] not in citing_sentences:
                    include = True
                    for w in mw:
                        if w.lower() not in d['current'].lower():
                            include = False
                            break
                    if include:
                        d['label'] = label
                        citing_sentences.add(d['current'])
                        booted_data.append(d)

    # Idea 3: If a citing sentence has more than 3 citations, these citations
    # are most likely Neut
    for d in dataset:
        citing_s = d['current']
        if citing_s not in citing_sentences and len(
                utils.get_markers(citing_s)) > 3:
            d['label'] = 'Neut'
            citing_sentences.add(citing_s)
            booted_data.append(d)

print 'Citation function data bootstrapped, %d instances' % (len(booted_data))
# data_func.save_data(booted_data, 'booted_func_data_func.json')

# # Store the list of articles used for bootstrapping
# path = config.ANNOTATION_ARTICLES_BOOTSTRAP_FUNC_FILEPATH
# with open(path, 'wb') as fp:
#     fp.write('\n'.join(articles))
