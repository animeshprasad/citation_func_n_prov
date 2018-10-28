import nltk
import re
import HTMLParser
import string
import unicodedata
import time
import numpy as np
import config.config as config
import lib.regex as regex
import config.params_func as params_func
import config.params_prov as params_prov
from lxml import etree
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.stem.porter import *

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def remove_escapechar(s):
    """
    Remove HTML and Unicode escaped characters.
    """
    s = unicodedata.normalize('NFD', unicode(s)).encode('ascii', 'ignore')
    s = HTMLParser.HTMLParser().unescape(s)
    return s


def convert_escapechar(s):
    """
    Convert common escape characters to a display-able format.
    # TODO Still have "\'" in the output string
    """
    s = s.replace(u'\u2013', u'-')
    s = s.replace(u'\u2014', u'-')
    s = s.replace(u'\u2018', u'\'')
    s = s.replace(u'\u2019', u'\'')
    s = s.replace(u'\u201c', u'"')
    s = s.replace(u'\u201d', u'"')
    s = s.replace(u'\u2032', u'\'')
    s = str(remove_escapechar(s))
    return s


def remove_whitespace(s):
    """
    Remove all white spaces in the given string, including spaces between words
    and newline characters.
    """
    return ''.join(s.split())


def remove_newline(s):
    """
    Remove only the newline characters of the given string.
    """
    s = s.replace('-\n', '')
    s = s.replace('\n', ' ')
    return s


def remove_punc(s):
    """
    Remove all punctuations in a given string.
    """
    return ''.join(w for w in s if w not in string.punctuation)


def remove_stop(s):
    """
    Remove all stopwords from a sentence.
    """
    s = s.lower()
    return ' '.join([w for w in s.split() if w not in stop_words])


def remove_decimals(s):
    """
    Remove all numbers/decimals from a sentence.
    """
    numbers = re.finditer(regex.DECIMAL, s)
    for n in numbers:
        s = s.replace(n.group(), '')
    return s


def remove_markers(context):
    """
    Remove all citation markers from the given context.
    """
    context = remove_newline(context)
    m1 = re.finditer(regex.CITATION_PATTERN_1, context)
    m2 = re.finditer(regex.CITATION_PATTERN_2, context)
    for m in m1:
        context = context.replace(m.group(), '')
    for m in m2:
        context = context.replace(m.group(), '')
    return context


def remove_special(s):
    """
    Remove all special, funny characters such as \x84.
    """
    printable = set(string.printable)
    return filter(lambda x: x in printable, s)


def process_sentence(s):
    """
    Performs the most frequent type of sentence cleaning in this project.
    Given a string, returns the processed list of words.
    """
    s = remove_markers(s)
    s = remove_punc(s)
    s = remove_decimals(s)
    s = s.split()
    s = map(lambda x: x.lower(), s)
    return s


def replace_char(txt):
    """
    Given a text snippet, replace all citation markers & numerical values by
    special tokens.
    """
    txt = remove_newline(txt)

    # Citation markers
    m1 = re.finditer(regex.CITATION_PATTERN_1, txt)
    for m in m1:
        txt = txt.replace(m.group(), config.VocabChar.MARKER_TOKEN.value)

    m2 = re.finditer(regex.CITATION_PATTERN_2, txt)
    for m in m2:
        txt = txt.replace(m.group(), config.VocabChar.MARKER_TOKEN.value)

    # Numerics
    decimals = re.finditer(regex.DECIMAL, txt)
    for d in decimals:
        txt = txt.replace(d.group(), config.VocabChar.DECIMAL_TOKEN.value)

    numbers = re.finditer(regex.NUMBER, txt)
    for n in numbers:
        txt = txt.replace(n.group(), config.VocabChar.NUMBER_TOKEN.value)

    return txt


def get_decimals(s):
    """
    Return all numbers/decimals from a sentence. Convert all numbers with
    percentage sign to decimal format.
    """
    numbers = list(re.finditer(regex.DECIMAL, s))
    numbers = map(lambda x: x.group(), numbers)
    for i, n in enumerate(numbers):
        if n.endswith('%'):
            percentage = float(n[:-1]) / 100.0
            numbers[i] = percentage
        else:
            numbers[i] = float(n)
    return numbers


def stem(word):
    """
    Perform Porter stemming on a given word.
    """
    return stemmer.stem(word)


def get_sentences(p):
    """
    Given a paragraph, get its sentences as a list.
    """
    # Process `et al.'; sort of hackish though...
    p = re.sub(r'et al\. ', 'et al ', p)
    p = re.sub(r'et al\. *\(', 'et al (', p)
    p = re.sub(r'et al\. *\[', 'et al [', p)
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    return tokenizer.tokenize(p)


def get_markers(s):
    """
    Return all citation markers within a given string. Converts markers of the
    form [3, 4] to [3] and [4].
    """
    # Matching citation marker patterns
    s = remove_newline(s)
    m1 = re.finditer(regex.CITATION_PATTERN_1, s)
    m2 = re.finditer(regex.CITATION_PATTERN_2, s)

    markers = []
    for m in m1:
        m = m.group()
        numbers = m[1:-1]
        if '-' in m:
            numbers = numbers.split('-')
            lower = int(numbers[0])
            upper = int(numbers[1])
            for n in range(lower, upper + 1):
                markers.append('[%d]' % n)
        elif ',' in m:
            numbers = numbers.split(',')
            for n in numbers:
                n = int(n.strip())
                markers.append('[%d]' % n)
        else:
            n = int(numbers)
            markers.append('[%d]' % n)
    for m in m2:
        markers.append(m.group())
    return markers


def letter2int(l):
    """
    Mapping of letters: a -> 1, b -> 2, etc.
    """
    assert len(l) == 1
    return ord(l[0]) - ord('a') + 1


def is_surname(s):
    """
    Given string, returns true if it is likely to be a surname.
    # TODO Is this repetitive with the author regex?
    """
    # TODO I think should delete this...
    # if s.lower() != 'he' and s.lower() in stop_words:
    # Start of sentence likely to be capitalized and in stop words
    # return False
    if s.lower() in ['section', 'chapter', 'figure', 'table', 'appendix',
                     'note']:
        return False
    surname = s[0].isupper()
    return surname


def is_word(w):
    """
    Given a word, return True if it is an English word.
    """
    return w in words.words()


def jaccard(a, b):
    """
    Calculate the Jaccard coefficient between two sets.
    """
    c = a.intersection(b)
    div = (len(a) + len(b) - len(c))
    if div != 0:
        return float(len(c)) / (len(a) + len(b) - len(c))
    else:
        return 1


def overlap(a, b):
    """
    Returns the overlap between two sets.
    """
    c = a.intersection(b)
    return len(c)


def get_random(ls, num):
    """
    Get num elements from list randomly.
    """
    indices = np.random.permutation(len(ls))
    indices = indices[0:num]
    res = []
    for i in indices:
        res.append(ls[i])
    return res


def get_mlsets_indices(length):
    """
    Return randomised train and test dataset indices.
    """
    split = 0.7
    split_len = int(split * length)

    indices = range(0, length)
    train_indices = get_random(indices, split_len)
    test_indices = [i for i in indices if i not in train_indices]
    return (train_indices, test_indices)


"""
Operations for sentences.
"""


def sent_add_tag(s):
    """
    Add <S></S> tag to the sentence.
    """
    return '%s%s%s' % ('<S>', s, '</S>')


def sent_remove_tag(s):
    """
    Remove <S></S> tag from the sentence.
    """
    starts = re.finditer(regex.SENT_START, s)
    for n in starts:
        s = s.replace(n.group(), '')
    ends = re.finditer(regex.SENT_END, s)
    for n in ends:
        s = s.replace(n.group(), '')
    return s

    # return etree.fromstring(s, parser=etree.XMLParser(
    #     recover=True)).text


def sents_remove_tag(s):
    """
    Remove tags from several sentences; return as an array.
    """
    starts = re.finditer(regex.SENT_START, s)
    for n in starts:
        s = s.replace(n.group(), '')
    ends = re.finditer(regex.SENT_END, s)
    for n in ends:
        s = s.replace(n.group(), ' ')
    return s

    # res = []
    # delimiter = '</S>'
    # print 'sentences'
    # print ss
    # for s in ss.split(delimiter):
    #     s = s.strip() + delimiter
    #     if len(s) <= len(delimiter):
    #         continue
    #     print s
    #     res.append(sent_remove_tag(s))
    # return res


timestamp = None


def get_timestamp():
    """
    Get current time stamp.
    """
    timestamp = int(round(time.time() * 1000))
    return timestamp


def get_last_timestamp():
    if timestamp == None:
        return get_timestamp()
    else:
        return timestamp
