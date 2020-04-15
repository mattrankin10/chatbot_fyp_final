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


loaded_model = tf.keras.models.load_model('text_classify_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

max_len = 14
txt='where is the football pitch?'
seq= loaded_tokenizer.texts_to_sequences([txt])
print(seq)
padded = pad_sequences(seq, maxlen=max_len, padding='post')
print(padded)
pred = loaded_model.predict(padded)
print(pred)