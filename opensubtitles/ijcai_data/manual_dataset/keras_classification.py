import keras as keras
import pandas
import numpy as np
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Dropout, Embedding, Flatten, Activation, BatchNormalization, MaxPooling1D, Convolution1D
from keras.layers import Input, Dense, Dropout, Flatten, Concatenate
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import itertools
from keras.models import Model
import pickle

profile = 'name'

dataframe = pandas.read_csv(profile + "_keras.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:, 0]
Y = dataset[:, 1]



# print(X)
# tokenize X
for message in X:
    tokens = word_tokenize(message)
    # stemming of words
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in tokens]
    print(stemmed[:100])


tokenizer = Tokenizer()
# tokenize and build vocab

word_index = tokenizer.word_index
# The first indices are reserved
word_index["<PAD>"] = 0

tokenizer.fit_on_texts(X)
vocab_size = len(tokenizer.word_index) + 1

x_train = tokenizer.texts_to_sequences(X)
max_sentence = max(len(l) for l in x_train)

# create 1 array from multiple arrays

train_data = pad_sequences(x_train,
                             value=word_index["<PAD>"],
                             padding='post',
                             maxlen=max_sentence)

print(train_data)

# encode document

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

print(encoded_Y)

EMBEDDING_DIM=100
filter_sizes = (2,4,5,8)
dropout_prob = [0.4,0.5]

# graph_in = Input(shape=(n_in, EMBEDDING_DIM))
# convs = []
# avgs = []
# for fsz in filter_sizes:
#     conv = Convolution1D(nb_filter=32,
#                          filter_length=fsz,
#                          border_mode='valid',
#                          activation='relu',
#                          subsample_length=1)(graph_in)
#     pool = MaxPooling1D(pool_length=n_in-fsz+1)(conv)
#     flattenMax = Flatten()(pool)
#     convs.append(flattenMax)
#
# if len(filter_sizes)>1:
#     out = Concatenate(mode='concat')(convs)
# else:
#     out = convs[0]
#
# graph = Model(input=graph_in, output=out, name="graphModel")
#sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

# baseline model

model = Sequential()
model.add(Embedding(vocab_size, 64, input_length=max_sentence))
model.add(Dense(64, input_dim=train_data.shape[1]))

model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_data, encoded_Y , validation_split=0.2, verbose=True, epochs=100, batch_size=10)

# sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
# kfold = StratifiedKFold(n_splits=10, shuffle=True)
# cvscores = []
# # baseline model
# for train, test in kfold.split(train_data,encoded_Y):
#     # create model
#     model = Sequential()
#     model.add(Embedding(vocab_size, 64, input_length=max_sentence))
#     model.add(Dense(60, input_dim=14))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Dropout(0.4))
#
#     model.add(Flatten())
#     model.add(Dense(1, activation='sigmoid'))
#     # Compile model
#     model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
#     # Fit the model
#     model.fit(train_data[train], encoded_Y[train] , epochs=150, batch_size=10, verbose=0,
#               validation_data=(train_data[test], encoded_Y[test]))
#     # evaluate the model
#     scores = model.evaluate(train_data[test], encoded_Y[test], verbose=0)
#     loss, accuracy = model.evaluate(train_data[train], encoded_Y[train], verbose=False)
#     print("Training Accuracy: {:.4f}".format(accuracy))
#     loss, accuracy = model.evaluate(train_data[test], encoded_Y[test], verbose=False)
#     print("Testing Accuracy:  {:.4f}".format(accuracy))
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
#     cvscores.append(scores[1] * 100)

#
# print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
# evaluate model with standardized dataset
print(model.summary())
model.save('text_classify_model.h5')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)




