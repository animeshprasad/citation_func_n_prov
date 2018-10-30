"""
Configuration file for the project.
"""

"""
Base directory.
"""
PWD = '/home/citation-analysis'

"""
File directories.
"""
# Directory for the word embeddings
GLOVE_DIR = PWD + '/glove.6B'

# Directory for storing citation function data
DATA_DIR = PWD + '/data/data'

"""
Data files: the citation and provenance dataset.
MTL refers to the aligned dataset.
"""
DATA_FILES = {
    'func': {
        'golden_train': 'processed/golden_train.func.json',
        'golden_test': 'processed/golden_test.func.json',
    },
    'prov': {
        'golden_train': 'processed/golden_train.prov.json',
        'golden_test': 'processed/golden_test.prov.json',
    },
    'mtl': {
        'golden_train': 'processed/golden_train.mtl.json',
        'golden_test': 'processed/golden_test.mtl.json'
    }
}
