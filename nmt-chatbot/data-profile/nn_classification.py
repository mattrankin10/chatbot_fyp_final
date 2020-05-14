import Levenshtein
import keras as keras
import pandas
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
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
import pickle
import tensorflow as tf

profiles = ['employer_new']
for profile in profiles:
    loaded_model = tf.keras.models.load_model(profile + '_2_classify_model.h5')

    with open('tokenizer_2' + profile + '.pickle', 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)

    dataframe = pandas.read_csv(profile + ".csv", header=None)
    dataset = dataframe.values
    # split into input (X) and output (Y) variables
    X = dataset[:, 0]
    x_train = loaded_tokenizer.texts_to_sequences(X)
    max_len = max(len(l) for l in x_train)
    txt = []
    answer = ["What is your favourite hobby?", "what do you like to do?", "Hey what's up"]

    with open('testset_extras.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            txt.append(line.replace('\n', ''))

    for seq in txt:
        sentence = loaded_tokenizer.texts_to_sequences([seq])
        padded = pad_sequences(sentence, maxlen=max_len, padding='post')
        pred = loaded_model.predict(padded)
        ratio = max([Levenshtein.ratio(seq, s) for s in answer])

        print(seq + ': ')
        print(pred)
        print('similarity ratio =' + str(ratio))