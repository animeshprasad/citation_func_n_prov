"""
Processing the parameter tuning experimental results. This code is sort of
dirty...
"""
import csv
import config.config as config
import numpy as np
import matplotlib.pyplot as pl

# Provenance paths only
provenance = True

if provenance:
    paths = [config.MODEL_EVAL_COMPLETE_FILEPATH % ('prov', '4.1'),
             config.MODEL_EVAL_COMPLETE_FILEPATH % ('prov', '4.2'),
             config.MODEL_EVAL_COMPLETE_FILEPATH % ('prov', '4.3'),
             config.MODEL_EVAL_COMPLETE_FILEPATH % ('prov', '4.4'),
             config.MODEL_EVAL_COMPLETE_FILEPATH % ('prov', '4.5'), ]
else:
    paths = [config.MODEL_EVAL_COMPLETE_FILEPATH % ('func', '6.1'),
             config.MODEL_EVAL_COMPLETE_FILEPATH % ('func', '6.2'),
             config.MODEL_EVAL_COMPLETE_FILEPATH % ('func', '6.3'),
             config.MODEL_EVAL_COMPLETE_FILEPATH % ('func', '6.4'),
             config.MODEL_EVAL_COMPLETE_FILEPATH % ('func', '6.5')]

# Experimented parameter values, with the first values being default
params_main = {
    'batch_size': [32, 4, 8, 32, 64, 128],
    'nb_epoch': [15, 10, 15, 20, 25, 30],
    'dropout_rate': [0.5, 0.2, 0.3, 0.4, 0.5, 0.6],
    'trainable': [False, False, True],
    'nb_words': [10000, 8000, 10000, 15000, 20000, 25000],
    'skip_top': [40, 0, 20, 30, 40, 50, 80],
    'maxlen': [64, 48, 64, 128, 192, 256, None],
    'embedding_dims': [200, 50, 100, 200, 300]
}

# Initializations
# E.g. 'cnn' => {'default' => [f1_1, f1_2], 'maxlen' => { 48 => [f1_3, f1_4] }}
evals = {}

if provenance:
    models = ['CNN', 'LSTM', 'BiLSTM', 'GRU']
    #           '2CNN', '2LSTM', '2BiLSTM', '2GRU']
    # models = ['2CNN', '2LSTM', '2BiLSTM', '2GRU']
else:
    models = ['CNN', 'LSTM', 'BiLSTM', 'GRU']

for m in models:
    m = m.lower()
    evals[m] = {'default': list()}

    for param in params_main:
        if param not in evals[m]:
            evals[m][param] = {}

        for value in params_main[param]:
            value = str(value).lower()
            if value not in evals[m][param]:
                evals[m][param][value] = list()

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
Aggregate parameter tuning results and perform averaging.
"""
for entry in results:
    if len(entry) == 0:
        continue
    model = entry[0].strip()
    if model not in map(lambda x: x.lower(), models):
        continue

    f1 = float(entry[-1].strip())

    num_unequal = 0  # Number of params not equal to default
    idx_unequal = 0  # Value of the unequal param
    for i in range(1, len(entry) - 3):
        param = headers[i]
        if entry[i].lower().strip() != str(
                params_main[param][0]).lower().strip():
            num_unequal += 1
            idx_unequal = i

    if num_unequal == 0:
        # All params are equal to default
        evals[model]['default'].append(f1)

        for param in headers[1:]:
            if param not in params_main:
                continue

            value = str(params_main[param][0]).lower()
            if len(value) == 0:
                value = 'none'
            evals[model][param][value].append(f1)
    elif num_unequal == 1:

        # The parameter that is not equal to default
        param = headers[idx_unequal].strip()
        value = str(entry[idx_unequal]).lower()
        if len(value) == 0:
            value = 'none'
        evals[model][param][value].append(f1)
    else:
        # Two or more not equal
        print 'Error: %s' % (entry)

print evals

# Getting average performance
for m in evals:
    for param in evals[m]:
        if param == 'default':
            evals[m][param] = np.mean(evals[m][param][0:len(paths)])
        else:
            for value in evals[m][param]:
                evals[m][param][value] = np.mean(
                    evals[m][param][value][0:len(paths)])

print evals

"""
Plotting graphs.
"""
# Example using param: 'nb_words'
colours = ['r', 'g', 'b', 'y', 'c', 'm', 'k', '0.75']
minx = 0
maxx = 0

param = 'batch_size'
param = 'nb_epoch'
param = 'dropout_rate'
param = 'trainable'
param = 'nb_words'
param = 'skip_top'
param = 'maxlen'
param = 'embedding_dims'


def proc_x(x):
    # Process x
    if param == 'dropout_rate':
        return float(x)
    if param == 'maxlen' and x == 'none':
        return 300
    if param == 'trainable':
        if x == 'true':
            return 1
        else:
            return 0
    return int(x)


def rev_proc_x(x):
    # Reverse process x
    if param == 'dropout_rate':
        return str(x)
    if param == 'maxlen' and x == 300:
        return 'none'
    if param == 'trainable':
        if x == 1:
            return 'true'
        else:
            return 'false'
    return str(x)


for i, m in enumerate(models):
    lower_m = m.lower()
    print evals[lower_m][param]

    mx = list(evals[lower_m][param].keys())
    mx = sorted(map(proc_x, mx))
    minx = min(mx)
    maxx = max(mx)

    my = list()
    for x in mx:
        my.append(evals[lower_m][param][rev_proc_x(x)])

    plot = pl.plot(mx, my, linestyle='-', color=colours[i], marker='o', label=m)

pl.title('F1 scores of models vs. %s' % (param))

pl.xlabel(param)
pl.ylabel('F1 scores')

# pl.ylim(0.55, 0.75)
if provenance:
    position = 'lower right'
else:
    position = 'upper right'

if param == 'batch_size':
    pl.xlim(2, 136)
    pl.legend(loc=position)
elif param == 'nb_epoch':
    pl.xlim(5, 35)
    pl.legend(loc=position)
elif param == 'dropout_rate':
    pl.xlim(0.1, 0.7)
    pl.legend(loc=position)
elif param == 'trainable':
    pl.xlim(0, 1)
    pl.legend(loc=position)
elif param == 'nb_words':
    pl.xlim(7000, 26000)
    pl.legend(loc=position)
elif param == 'skip_top':
    pl.xlim(-5, 90)
    pl.legend(loc=position)
elif param == 'maxlen':
    pl.xlim(40, 308)
    pl.legend(loc=position)
elif param == 'embedding_dims':
    pl.xlim(45, 305)
    pl.legend(loc=position)

pl.show()
