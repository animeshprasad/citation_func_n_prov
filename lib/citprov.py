"""
This file defines operations related to citation provenance.
"""
import random

provs = {
    'Prov': 0,
    'Non-Prov': 1
}
inv_provs = {v: k for k, v in provs.items()}
count = len(provs)


def prov2int(prov):
    return provs[prov]


def int2prov(i):
    return inv_provs[i]
