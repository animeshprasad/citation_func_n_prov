"""
This file defines operations related to citation functions.
"""
import random

funcs = {
    'Weak': 0,
    'CoCo': 1,
    'Pos': 2,
    'Neut': 3
}
inv_funcs = {v: k for k, v in funcs.items()}
count = len(funcs)


def func2int(func):
    return funcs[func]


def int2func(i):
    return inv_funcs[i]
