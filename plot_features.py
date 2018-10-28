"""
A script for plotting / tabulating the performance of models with handcrafted
features and without.
"""
import csv
import sys
import config.config as config
import numpy as np
import matplotlib.pyplot as pl

paths = [config.MODEL_EVAL_FILEPATH % ('func', '7.1')]

# Initializations
# E.g. 'cnn' => { 0: { 0: { 'precision': [], 'recall': [] }, 1: {...} } }
# Outer 0: features, inner 0: classes
evals = {}
models = ['CNN', 'BiLSTM']

for m in models:
    m = m.lower()
    evals[m] = {}

    for f in range(0, 7):
        evals[m][f] = {}
        evals[m][f]['precision'] = list()
        evals[m][f]['recall'] = list()

        for c in range(0, 4):
            evals[m][f][c] = {}

            evals[m][f][c]['precision'] = list()
            evals[m][f][c]['recall'] = list()

print evals

# Read data from all files
all_data = list()
for path in paths:
    f = open(path, 'rb')
    f_data = list(csv.reader(f))
    f.close()
    all_data += f_data

headers = all_data[0]
results = all_data[1:]

"""
Aggregate evaluation results and perform averaging.
"""
for entry in results:
    if len(entry) == 0:
        continue
    model = entry[0].strip()
    if model not in map(lambda x: x.lower(), models):
        continue

    if len(entry[1]) < 1:
        # 0: No feature used
        f = 0
    else:
        f = int(entry[1].strip()) + 1

    evals[model][f]['precision'].append(float(entry[-3].strip()))
    evals[model][f]['recall'].append(float(entry[-3].strip()))

    for c in range(0, 4):
        precision = float(entry[2 + 3 * c].strip())
        recall = float(entry[3 + 3 * c].strip())
        evals[model][f][c]['precision'].append(precision)
        evals[model][f][c]['recall'].append(recall)

print evals

# Getting average performance
for m in evals:
    for f in evals[m]:
        evals[m][f]['precision'] = np.mean(evals[m][f]['precision']) if len(
            evals[m][f]['precision']) > 0 else 0
        evals[m][f]['recall'] = np.mean(evals[m][f]['recall']) if len(
            evals[m][f]['recall']) > 0 else 0

        for c in range(0, 4):
            evals[m][f][c]['precision'] = np.mean(
                evals[m][f][c]['precision']) if len(
                evals[m][f][c]['precision']) > 0 else 0
            evals[m][f][c]['recall'] = np.mean(evals[m][f][c]['recall']) if len(
                evals[m][f][c]['recall']) > 0 else 0

print evals

"""
Plotting graphs.
"""
colours = ['r', 'g', 'b', 'y', 'c', 'm', 'k', '0.75']

for i in range(0, 7):
    lower_m = 'bilstm'

    mx = [0, 1, 2, 3, 4]

    my = list()
    for x in mx[:-1]:
        precision = evals[lower_m][i][x]['precision']
        recall = evals[lower_m][i][x]['recall']
        if precision != 0 and recall != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        my.append(f1)

    precision = evals[lower_m][i]['precision']
    recall = evals[lower_m][i]['recall']
    if precision != 0 and recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    print f1
    my.append(f1)

    plot = pl.plot(mx, my, linestyle='-', color=colours[i], marker='o',
                   label='CNN: %s' % (i))

pl.title('F1 scores of models vs. selected feature')

pl.xlabel('Class')
pl.ylabel('F1 scores')

pl.xlim(-0.5, 4.5)
pl.legend(loc='upper left')

pl.show()
