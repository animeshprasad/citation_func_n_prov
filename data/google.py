"""
This script is in charge of interacting with Google APIs. Importantly, it
performs sentiment analysis.
"""
# Google Cloud uses new Python namespace format..
from __future__ import absolute_import

import numpy as np
from config import google_credentials
from google.cloud import language
from lib import citfunc

language_client = language.Client()


def get_credentials():
    return {
        'type': google_credentials.TYPE,
        'project_id': google_credentials.PROJECT_ID,
        'private_key_id': google_credentials.PRIVATE_KEY_ID,
        'private_key': google_credentials.PRIVATE_KEY,
        'client_email': google_credentials.CLIENT_EMAIL,
        'client_id': google_credentials.CLIENT_ID,
        'auth_uri': google_credentials.AUTH_URI,
        'token_uri': google_credentials.TOKEN_URI,
        'auth_provider_x509_cert_url': google_credentials.AUTH_PROVIDER_CERT_URL,
        'client_x509_cert_url': google_credentials.CLIENT_CERT_URL
    }


def print_result(annotations):
    score = annotations.sentiment.score
    magnitude = annotations.sentiment.magnitude

    for index, sentence in enumerate(annotations.sentences):
        sentence_sentiment = sentence.sentiment.score
        print('Sentence {} has a sentiment score of {}'.format(
            index, sentence_sentiment))

    print('Overall Sentiment: score of {} with magnitude of {}'.format(
        score, magnitude))
    return 0


def get_sentiment(sentence):
    """
    Obtain sentiment values of a single sentence.
    """
    document = language_client.document_from_text(sentence)
    annotations = document.annotate_text(include_sentiment=True,
                                         include_syntax=False,
                                         include_entities=False)
    print_result(annotations)
    return annotations


def get_func_sentiments(dataset):
    """
    Given a citation function dataset, print the aggregated sentiment values.
    """
    classes = citfunc.funcs
    for k in classes.keys():
        instances = filter(lambda x: x['label'] == k, dataset)
        sentiments = [get_sentiment(s['current']) for s in instances]
        scores = [s.score for s in sentiments]
        magnitudes = [s.magnitude for s in sentiments]

        meanscore = np.mean(scores)
        stdscore = np.std(scores)

        print 'Class: {}'.format(k)
        print 'Mean: {}'.format(meanscore)
        print 'Standard deviation: {}'.format(stdscore)
