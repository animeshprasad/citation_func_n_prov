"""
This file handles all activities related to parsing the paper using ParsCit,
including fetching relevant information from the paper.
"""

import bibtexparser
import csv
import os
import operator
import json
import hashlib
import random
import re
import time
import xml.etree.ElementTree as ET
import xmldict
import config.config as config
import config.exceptions as exceptions
import lib.file_ops as file_ops
import lib.regex as regex
import lib.utils as utils
from bs4 import BeautifulSoup
from lxml import etree
from urllib2 import Request, urlopen, quote

# --------------- Functions for getting file names and paths ---------------
"""
In the following, article refers to article ID, such as A83-1023.
"""


def get_parsed_path(article):
    return os.path.join(
        config.ACL_ARC_DIR + '/%s/%s' % (article[0], article[0:3]),
        article + '-parscit.130908' + file_ops.XML_EXTENSION)


def get_metadata_path(article):
    """
    Metadata is a file storing relevant information about the given article. The
    info includes official Bib information and information about the references.
    """
    return os.path.join(
        config.ACL_ARC_DIR + '/%s/%s' % (article[0], article[0:3]),
        article + '-parscit.130908' + file_ops.XML_EXTENSION + '.meta')


# --------------- END Functions for getting file names and paths ---------------


# --------------- Functions for parsing and generating metadata ---------------

# Predicates for checking whether the files have been generated
def is_parsed(article):
    """
    Given the ID of an article, return true if it has been parsed using ParsCit,
    false otherwise.
    """
    parsed_path = get_parsed_path(article)
    return file_ops.file_exists(parsed_path)


def has_metadata(article):
    """
    Returns true if the metadata for the given file has been generated.
    """
    metadata_path = get_metadata_path(article)
    return file_ops.file_exists(metadata_path)


# Actual parsing / generation
def parse(article):
    """
    Parse the article with the given name. Returns the parsed string.
    - Assumption: All files in the ACL ARC have been parsed, so this method is
    obsolete
    """
    if not is_parsed(article):
        raise Exception(exceptions.FILE_NOT_PARSED_EXCEPTION)
    filepath = get_parsed_path(article)
    parsed = ET.parse(filepath, parser=etree.XMLParser(recover=True)).getroot()
    return ET.tostring(parsed)


def get_reference_indices(article):
    """
    Return all reference indices.
    """
    metadata = get_metadata(article)
    return metadata['indices']


def get_reference_index(marker, context, article):
    """
    Given the marker, context, and article, decide which reference does the
    marker refer to in the `references' section. Index starts counting from 1.
    # Marker pattern 1: assume is of the format [4], i.e. no '-' or ','
    # Marker pattern 2: can be like Teuffel, 2006b; including the letter
    # FINISHED, I think still BUGGY
    """
    # Matching citation marker patterns
    m1 = re.search(regex.CITATION_PATTERN_1, marker)
    m2 = re.search(regex.CITATION_PATTERN_2, marker)

    if m1:
        # Trivial
        m1 = m1.group()
        return int(m1[1:-1])
    if m2:
        # Need to process the marker, and search it within references
        # Use year, author name, and context as clues
        year = re.search(regex.CITATION_YEAR, marker).group()
        if year[-1].isalpha():
            # Year contains a letter
            year_index = utils.letter2int(year[-1])
            year = int(year[:-1])
        else:
            year_index = 1
            year = int(year)
        # print "Marker: %s" % (marker)
        author = re.search(regex.CITATION_AUTHOR, marker)
        if author and len(author.group()) > 1:
            author = utils.remove_punc(author.group())
            author = filter(utils.is_surname, author.split())
        else:
            # Else: author needs to be parsed from the citing sentence
            author = utils.remove_punc(context[1])
            author = filter(utils.is_surname, author.split())

        # Determine which reference is being referred to
        parsed = ET.fromstring(parse(article),
                               parser=etree.XMLParser(recover=True))
        citation_list = list(parsed.iter('citationList').next())

        # List of possible references
        candidates = {}

        for i, c in enumerate(citation_list):
            c_year = list(c.iter('date'))
            if len(c_year) == 0 or c_year[0].text == None or len(
                    c_year[0].text) < 4:
                # TODO This seems to have some bug on E06-1050.pdf; once of the
                # dates not read
                continue
            c_year = int(c_year[0].text.strip())

            if c_year == year:
                # This comparison is using only surname because that's how
                # citation markers are
                c_authors = list(c.iter('author'))
                c_authors = map(lambda a:
                                utils.remove_punc(utils.remove_special(
                                    a.text.split(' ')[-1].lower())),
                                c_authors)
                # print 'author'
                # print author
                # print 'c_authors'
                # print c_authors

                # Go through each of the authors and compare
                i_1 = 0
                i_2 = 0
                overlap = 0
                while i_1 < len(author) and i_2 < len(c_authors):
                    a_1 = author[i_1].lower()
                    if a_1 in c_authors:
                        i_1 += 1
                        i_2 = c_authors.index(a_1) + 1
                        overlap += 1
                    else:
                        i_1 += 1
                if overlap > 0:
                    candidates[i + 1] = -overlap
                    # print overlap

        # TODO This step may have bugs
        candidates = sorted(candidates.items(), key=operator.itemgetter(1))
        candidates = list(c[0] for c in candidates)
        if len(candidates) >= 1:
            # To ensure no bug occurs, use min
            return candidates[min(year_index, len(candidates)) - 1]
        else:
            print 'Marker: %s' % (marker)
            print 'Context: %s' % (context)
            print 'Article: %s' % (article)
            print 'ERROR!'
            return -1


def get_bib_info(article):
    """
    Get the official BibTex citation information of the article.
    """
    try:
        # Try first link
        link = 'https://aclweb.org/anthology/%s/%s/%s.bib' % (
            article[0], article[0:3], article)
        print link
        bibtex = urlopen(Request(link)).read()
        bibtex = bibtexparser.loads(bibtex)
        return bibtex.entries[0]
    except Exception, e:
        try:
            # Try second link
            link = 'http://www.aclweb.org/anthology/%s.bib' % (article)
            print link
            bibtex = urlopen(Request(link)).read()
            bibtex = bibtexparser.loads(bibtex)
            return bibtex.entries[0]
        except Exception, e:
            # TODO
            return {}

            # Use Google Scholar then...
            print 'Error: Link not found!'
            time.sleep(16)

            # Parse to get article title first; used later for gscholar search
            parsed = ET.fromstring(parse(article),
                                   parser=etree.XMLParser(recover=True))
            title = list(parsed.iter('title'))
            title = title[0].text.strip().replace(' ', '+')

            # Generate random string to fake Google ID...
            rand_str = str(random.random()).encode('utf8')
            gid = hashlib.md5(rand_str).hexdigest()[:16]
            header = {'User-Agent': 'Mozzila/5.0',
                      'Cookie': 'GSP=ID=%s:CF=4' % gid}

            url = 'https://scholar.google.com.sg/scholar?hl=en&q=%s' % quote(
                title.encode('UTF-8'))
            print url
            request = Request(url, headers=header)
            response = urlopen(request)
            html = response.read()
            # print html
            soup = BeautifulSoup(html, 'lxml')

            # Get information from BibTex
            link = ''
            for tag in soup.find_all(
                    lambda tag: (
                                    tag.name == 'a' and tag.text == 'Import into BibTeX'),
                    href=True):
                link = tag['href']
                request = Request(link, headers=header)
                response = urlopen(request)
                bibtex = response.read()
                bibtex = bibtexparser.loads(bibtex)
                return bibtex.entries[0]

            return {}


def get_metadata(article):
    """
    Returns the metadata of the given article as a JSON dict. Metadata consists
    of: bib information, citation counts of the references, etc.
    # FINISHED
    """
    if has_metadata(article):
        filepath = get_metadata_path(article)
        with open(filepath, 'rb') as fp:
            metadata = json.load(fp)
            return metadata
    else:
        # Get data from official BibTex
        bib_info = get_bib_info(article)

        # Next, gather citation count information
        parsed = ET.fromstring(parse(article),
                               parser=etree.XMLParser(recover=True))
        references = list(parsed.iter('citationList').next())
        counts = dict()

        sentences = get_texts(article)
        for i, s in enumerate(sentences):
            markers = utils.get_markers(s)
            for m in markers:
                prev = sentences[i - 1] if i > 0 else ''
                next = sentences[i + 1] if i < len(sentences) - 1 else ''
                context = [prev, s, next]

                index = get_reference_index(m, context, article)
                if index not in counts:
                    counts[index] = 0
                counts[index] += 1
        counts = dict([k, v] for k, v in counts.iteritems())

        # Store reference indices
        indices = counts.keys()

        # Next, gather citation authors (both first name and last name)
        # - Format: F Surname
        authors = dict()
        for i, r in enumerate(references):
            r_authors = list(r.iter('author'))
            r_authors = map(lambda a: a.text, r_authors)
            if (i + 1) in indices:
                authors[(i + 1)] = r_authors

        # Finally, store to file
        metadata = bib_info
        metadata['citationCounts'] = counts
        metadata['indices'] = indices
        metadata['authors'] = authors
        filepath = get_metadata_path(article)

        with open(filepath, 'wb') as fp:
            json.dump(metadata, fp)
            return metadata


# --------------- END Functions for parsing and generating metadata ---------------


# --------------- Helper functions: get all body texts ---------------
def get_texts(article):
    """
    Get all texts of an article as a list of strings. Each element
    in this list is a sentence.
    # FINISHED
    """
    parsed = ET.fromstring(parse(article), parser=etree.XMLParser(recover=True))
    parsed = parsed.xpath('algorithm/variant')[0]
    texts = parsed.xpath('*/text()')

    res = []
    for t in texts:
        t_new = utils.remove_newline(t).strip()
        t_new = utils.remove_special(t_new)
        t_new = str(utils.convert_escapechar(t_new))
        if t_new.lower() == 'references':
            break
        if len(t_new) > 0:
            for s in utils.get_sentences(t_new):
                res.append(s)
    return res


# --------------- END Helper functions: get all body texts ---------------


# --------------- Functions for querying articles to get features ---------------
def get_year(article):
    """
    Get the publishing year of the citing article. Search in the collated
    article year file. The 3rd and 4th digits represent the year
    """
    year = article[1] + article[2]
    if article[1] in ['0', '1']:
        year = '20' + year
    else:
        year = '19' + year
    return int(year)

    # TODO The following is obsolete
    year_path = config.ANNOTATION_ARTICLES_YEARS_FILEPATH
    with open(year_path, 'rb') as f:
            results = list(csv.reader(f))
            for res in results:
                if res[0].strip() == article.strip():
                    return int(res[1].strip())
    return None

    # TODO The following is also obsolete
    metadata = get_metadata(article)
    if 'year' in metadata:
        year = int(metadata['year'])
        return year
    else:
        # Return -1 to indicate year not found
        return -1


def get_reference_year(article, reference_index):
    """
    Get the publishing year of the i-th reference in the article, where i starts
    from 1. If it's not there, default to 2011.
    """
    parsed = ET.fromstring(parse(article), parser=etree.XMLParser(recover=True))
    citation_list = list(parsed.iter('citationList').next())
    citation = citation_list[min(reference_index, len(citation_list)) - 1]

    year = list(citation.iter('date'))
    if len(year) > 0 and year[0].text != None:
        return int(year[0].text)
    else:
        return -1


def get_location(article, sentence):
    """
    Return the location of a given sentence within the paper. This location is
    represented as a number in [0, 1].
    # TODO This is a bit slow
    """
    parsed = ET.fromstring(parse(article), parser=etree.XMLParser(recover=True))

    prev_length = 0  # Lengths before the sentence
    sent_length = 0
    total_length = 0

    sentences = get_texts(article)
    sentences = map(utils.remove_escapechar, sentences)
    sentences = map(utils.remove_markers, sentences)
    sentences = map(utils.remove_punc, sentences)

    sentence = utils.remove_escapechar(sentence)
    sentence = utils.remove_markers(sentence)
    sentence = utils.remove_punc(sentence)
    words = set(sentence.split())

    index = 0  # Index for the sentence in body text
    jaccard = 0
    # Take the sentence with the most overlaps with the target sentence
    for i, s in enumerate(sentences):
        words_2 = set(s.split())
        j = utils.jaccard(words, words_2)
        if j > jaccard:
            jaccard = j
            index = i
        total_length += len(s)

    sent_length = len(sentence) / 2
    prev_length = len(''.join(sentences[:(index - 1)]))
    return float(prev_length + sent_length) / float(total_length)


def get_authors(article):
    """
    Get the authors of a citing article. Use BibTex info because results by
    ParsCit are not very satisfactory.
    - Format: F Surname
    """
    info = get_metadata(article)
    if 'author' in info:
        author = info['author'].split('and')
        authors = []
        for a in author:
            # Reformatting the name to make it match ParsCit output
            if ',' in a:
                names = a.split(',')
                surname = names[0].strip()
            else:
                names = a.strip().split(' ')
                surname = names[-1].strip()

            if len(names) > 1:
                if ',' in a:
                    firstname = names[1].replace('-', ' ').split()
                else:
                    firstname = ' '.join(names[0:-1]).replace('-', ' ').split()
                for i, w in enumerate(firstname):
                    firstname[i] = w[0].upper()
                firstname = ' '.join(firstname)
                authors.append('%s %s' % (firstname, surname))
            else:
                authors.append('%s' % (surname))
        return authors
    else:
        return []


def get_reference_authors(article, reference_index):
    """
    Get the authors of the i-th reference.
    - Format: F Surname
    """
    metadata = get_metadata(article)
    if str(reference_index) in metadata['authors']:
        return metadata['authors'][str(reference_index)]
    else:
        return []


def get_citation_count(article, index):
    """
    Return the citation count of the i-th reference in the citing paper.
    """
    metadata = get_metadata(article)
    if str(index) in metadata['citationCounts']:
        return metadata['citationCounts'][str(index)]
    else:
        return 0


# --------------- Functions for citation provenance ---------------
def get_fragments(article):
    """
    Given a cited article, return all fragments. First fragment is article
    abstract.
    """
    texts = get_texts(article)
    fragments = []

    # First fragment: Abstract
    abs_start = 0
    abs_end = 0
    for i, t in enumerate(texts):
        t = t.lower().strip()
        if t == 'abstract' or t.endswith('abstract') or t.startswith(
                'abstract'):
            abs_start = i
        if t == 'introduction' or t.endswith('introduction') or t.startswith(
                'introduction'):
            abs_end = i
            break
    abstract = ' '.join(texts[abs_start:abs_end])
    fragments.append(abstract)
    print abstract

    # Adding more fragments
    num_sents = config.NUM_SENTENCES_PER_FRAGMENT
    num_skips = config.NUM_SKIPS_PER_FRAGMENT

    index = abs_end
    while index < len(texts):
        new_fragment = texts[index:(index + num_sents)]
        new_fragment = ' '.join(new_fragment)
        fragments.append(new_fragment)
        index += num_skips

    return fragments
