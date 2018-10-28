"""
Configuration file for the project.
"""
from enum import Enum


"""
Base directory.
"""
PWD = '/Users/suxuan/Development/citation-analysis'

"""
Miscellaneous parameters.
"""
# Number of sentences preceding the citing sentence
NUM_SENTENCES_PRECEDING_CITING = 1

# Number of sentences in each fragment used for provenance
NUM_SENTENCES_PER_FRAGMENT = 5

# Number of skips per fragment; i.e. distance between current and previous fragments
NUM_SKIPS_PER_FRAGMENT = 2

# Base filter used for Keras tokenizers
BASE_FILTER = '!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n'  # So hackish...

"""
File directories.
"""
# Directory for the word embeddings
GLOVE_DIR = PWD + '/glove.6B'

# Directory for storing citation function data
DATA_DIR = PWD + '/data/data'

# Directories for storing SciSumm i.e. provenance data
SCISUMM_DEV_DIR = PWD + '/scisumm-corpus/data/Development-Set-Apr8'
SCISUMM_TEST_DIR = PWD + '/scisumm-corpus/data/Test-Set-2016'
SCISUMM_TRAIN_DIR = PWD + '/scisumm-corpus/data/Training-Set-2016'

# Directory for ACL Anthology parsed files
ACL_ARC_DIR = PWD + '/acl-arc'

"""
Specific file locations.
"""
# Articles, data instances used for the annotation system
ANNOTATION_ARTICLES_FILEPATH = PWD + '/data/data/citations_articles.txt'
ANNOTATION_FILEPATH = PWD + '/data/data/citations.txt'

"""
I have conducted a few rounds of annotations using MTurk and CrowdFlower.
Below are the files that store 1) the articles used, 2) the actual data
instances.
"""
# Articles & data instances for 1st round of annotation in CrowdFlower
ANNOTATION_MTURK_ARTICLES_FILEPATH_1 = PWD + '/data/data/citations_articles_mturk_1.txt'
ANNOTATION_MTURK_FILEPATH_1 = PWD + '/data/data/citations_1.csv'

# Articles & data instances for 2nd round of annotation in CrowdFlower
ANNOTATION_MTURK_ARTICLES_FILEPATH_2 = PWD + '/data/data/citations_articles_mturk_2.txt'
ANNOTATION_MTURK_FILEPATH_2 = PWD + '/data/data/citations_2.csv'

# Pilot articles used for MTurk
ANNOTATION_MTURK_ARTICLES_SMALL_FILEPATH = PWD + '/data/data/citations_articles_mturk_small.txt'

# Articles that are selected but not used in the annotation system (may be used in MTurk)
ANNOTATION_ARTICLES_TODELETE_FILEPATH = PWD + '/data/data/citations_articles_todelete.txt'

# Articles used for function bootstrapping
ANNOTATION_ARTICLES_BOOTSTRAP_FUNC_FILEPATH = PWD + '/data/data/citations_articles_bootstrap_func.txt'

# Articles and years
ANNOTATION_ARTICLES_YEARS_FILEPATH = PWD + '/data/data/citations_articles_year.csv'

"""
CitProv similarity bootstrapping, model and weight path. Using semeval2016.
"""
SEMEVAL_DIR = PWD + '/semeval2016'
SEMEVAL_MODEL_DIR = DATA_DIR + '/semeval_btsp'

# Path to word_index, storing word-index mappings
WORD_INDEX_FILEPATH = DATA_DIR + '/word_index.json'

# Path to the csv file storing model performance
# First %s: func or prov; second: experiment ID
MODEL_EVAL_FILEPATH = DATA_DIR + '/%s.eval.exp%s.csv'

# Complete path: All experiments here are completed
MODEL_EVAL_COMPLETE_FILEPATH = DATA_DIR + '/experiments/%s.eval.exp%s.csv'

# Func or prov; timestamp; model name; params; experiment serial number
MODEL_RES_FILEPATH = DATA_DIR + '/results/%s.%s.res.%s.%s.%s.csv'

CV_RES_FILEPATH = DATA_DIR + '/cv/results.csv'

# In data.py, whether to use class weights or not
USE_CLASS_WEIGHTS = True

# For citation function, whether to use only the citing sentence or not
USE_ONLY_CITING_SENTENCE = True

"""
Data files: the citation and provenance dataset.
MTL refers to the aligned dataset.
"""
DATA_FILES = {
    'func': {
        'golden_train': 'processed/golden_train.func.json',
        'golden_test': 'processed/golden_test.func.json',
        'silver': 'processed/silver.func.json'
    },
    'prov': {
        'golden_train': 'processed/golden_train.prov.json',
        'golden_test': 'processed/golden_test.prov.json',
        'silver': 'processed/silver.prov.json'
    },
    'mtl': {
        'golden_train': 'processed/golden_train.mtl.json',
        'golden_test': 'processed/golden_test.mtl.json'
    }
}

"""
Enumeration for special characters in the vocabulary.
"""


class VocabChar(Enum):
    # The start of a sequence will be marked with this character
    START = 1

    # Words that do not appear in word_index
    OOV = 2

    # By convention, use 2 as OOV word
    # Reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    # Real word index starts from 3, so need to offset by 2
    INDEX_OFFSET = 2

    # Citation markers
    MARKER = 3
    MARKER_TOKEN = str(' __MARKER__ ')

    # Numerical values
    DECIMAL = 4
    DECIMAL_TOKEN = str(' __DECIMAL__ ')

    NUMBER = 5
    NUMBER_TOKEN = str(' __NUMBER__ ')
