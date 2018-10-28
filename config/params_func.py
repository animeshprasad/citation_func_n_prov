"""
Common parameters for the CitFunc models / baselines.
"""

# Cross validation fold
cv_fold = 5

# max number of words to include. Words are ranked by how often they occur (in
# the training set) and only the most frequent words are kept
nb_words = 10000

# Skip the top N most frequently occuring words which may not be informative
skip_top = 40

# Max sequence length; used to pad sequences
maxlen = 64

# The size of each batch
batch_size = 32

# Dimension of the word vectors; can be 50,100,200,300 in this example
embedding_dims = 200

# Dimensions of the hidden units
hidden_dims = 100

# Number of epochs
nb_epoch = 15

# Learning rate
lr = 0.01

# Numpy seed for randomisation
seed = 683

# Dropout rate
dropout_rate = 0.5

# Whether the word embeddings are trainable or not
trainable = False

# Dimensionality reduction used in baseline, reduce to 300 dimensions
feature_size = 300

"""
CNN parameters.
"""
# Number of filters in convolutional models
nb_filter = 16

# The length of filters in convolutional models
filter_length = 3

"""
RNN model parameters.
"""
# Dimensionality of the LSTM output space
rnn_units = 64
