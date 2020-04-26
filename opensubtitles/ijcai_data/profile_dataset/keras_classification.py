import keras as keras
import pandas
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, AveragePooling1D, Flatten, Bidirectional, BatchNormalization, LSTM,\
    Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.text import Tokenizer
from keras import optimizers, regularizers
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import itertools
import pickle
from sklearn.model_selection import train_test_split


from tensorflow.python.client import device_lib


print(device_lib.list_local_devices())

profiles = ['hobby_new']

for profile in profiles:
    dataframe = pandas.read_csv(profile + ".csv", header=None)
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


    # encode document

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    print(encoded_Y)

    # split train and test data
    x_train, x_valid, y_train, y_valid = train_test_split(train_data, encoded_Y, test_size=0.33, shuffle=True)

    sgd = optimizers.SGD(lr=0.1)

    # baseline model

    model = Sequential()

    model.add(Embedding(vocab_size, 64, input_length=max_sentence))

    model.add(Dense(64, input_dim=train_data.shape[1]))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.8))

    model.add(Flatten())
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.4))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train , validation_data=(x_valid, y_valid), verbose=True, epochs=20, batch_size=16)


    print(model.summary())
    model.save(profile + '_classify_model.h5')
    with open('tokenizer_' + profile + '.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)




