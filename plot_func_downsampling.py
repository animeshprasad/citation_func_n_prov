"""
Draw plots of the downsampling experiments.
"""
import csv
import config.config as config
import numpy as np
import matplotlib.pyplot as pl

# Complete downsampling experiments
downsampling = True

if downsampling:
    paths = [config.MODEL_EVAL_COMPLETE_FILEPATH % ('prov', '5.1'),]
            # config.MODEL_EVAL_COMPLETE_FILEPATH % ('func', '5.2')]
    paths = [config.MODEL_EVAL_COMPLETE_FILEPATH % ('func', '10')]
    paths = [config.MODEL_EVAL_COMPLETE_FILEPATH % ('prov', '10')]
else:
    paths = [config.MODEL_EVAL_COMPLETE_FILEPATH % ('func', '2.1'),
            config.MODEL_EVAL_COMPLETE_FILEPATH % ('func', '2.2')]

if downsampling:
    alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
else:
    alphas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
    # alphas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

evals = {}

models = ['2CNN']
# models = ['CNN']

# Initialization
for m in models:
    m = m.lower()
    evals[m] = {}

    for alpha in alphas:
        evals[m][alpha] = list()

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
Aggregate the results and calculate errors.
"""
for entry in results:
    model = entry[0].strip().lower()
    if model not in evals:
        continue

    alpha = entry[1].strip()
    if downsampling:
        alpha = float(alpha)
    else:
        alpha = int(alpha)

    f1 = float(entry[-1].strip())
    if alpha in alphas:
        evals[model][alpha].append(f1)

""" 
Start to populate the plot.
"""
colours = ['r', 'g', 'b', 'y', 'c', 'm', 'k', '0.75']

for i, m in enumerate(models):
    lower_m = m.lower()
    # if i > 0:
    #     break

    mx = sorted(alphas)
    my = []
    my_err_up = []
    my_err_down = []

    for x in alphas:
        ys = evals[lower_m][x]
        ys_mean = np.mean(ys)
        ys_min = ys_mean - min(ys)
        ys_max = max(ys) - ys_mean

        my.append(ys_mean)
        my_err_up.append(ys_max)
        my_err_down.append(ys_min)

    plot = pl.errorbar(mx, my, yerr=[my_err_down, my_err_up], color=colours[i],
                       marker='o', label=m)
    # plot = pl.plot(mx, my, linestyle='-', color=colours[i], marker='o', label=m)

if downsampling:
    mx = [-0.05, 1.10]
else:
    mx = [-0.05, 31.05]

my = np.mean([0.6591, 0.6497, 0.6505, 0.6507, 0.6601, 0.6583])
my = np.mean([0.808, 0.812, 0.806])
my = [my, my]
pl.plot(mx, my, linestyle='-', color='0.2', marker='.', label='2CNN w/o silver')

# my = [0.69, 0.69]
# pl.plot(mx, my, linestyle='-', color='0.50', marker='.', label='BiLSTM w/o silver')

pl.title('F1 scores of models vs. silver data alpha value')

pl.xlabel('Alpha value')
pl.ylabel('F1 scores')

if downsampling:
    pl.xlim(0, 1.05)
else:
    pl.xlim(0.5, 30.5)

pl.legend(loc='upper right')

pl.show()
