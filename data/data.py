"""
Common data operations.
"""
import json
import os

import numpy as np

import config.config as config


def read_json_data(filename):
    """
    Read the given JSON file.
    """
    path = os.path.join(config.DATA_DIR, filename)
    with open(path, 'rb') as fp:
        content = json.load(fp)
        return content


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
