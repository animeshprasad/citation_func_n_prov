"""
This file defines modes for the citprov classifiers.
"""
# BTSP: Bootstrapping
MODE_SEMEVAL_BTSP = 0
MODE_USING_BTSP = 1
MODE_USING_BTSP_WO_TRAIN = 2  # Using bootstrapped model without further training
MODE_WITHOUT_BTSP = 3

MODES = {}


def mode2str(m):
    if len(MODES) == 0:
        MODES[0] = 'MODE_SEMEVAL_BTSP'
        MODES[1] = 'MODE_USING_BTSP'
        MODES[2] = 'MODE_USING_BTSP_WO_TRAIN'
        MODES[3] = 'MODE_WITHOUT_BTSP'
    return MODES[m]
