from __future__ import absolute_import, division, print_function, unicode_literals
from keras.preprocessing import sequence
import tensorflow_datasets as tfds
from tensorflow import TensorShape
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras import layers , activations , models , preprocessing , utils
from tensorflow.keras.layers import Input, GRU, Dense, Concatenate, TimeDistributed, Embedding, LSTM
from keras import layers, activations, models, preprocessing, utils
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
import re
import numpy as np
import tensorflow as tf
import os


def clean_line(line):
    return line.replace("\n", "").replace("\r", "")


def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    # adding a start and an end token to the sentence
    return sentence



def load_conversations():
    inputs = []
    outputs = []
    with open("train.from", 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        inputs.append(clean_line(line))

    with open("train.to", 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        outputs.append(clean_line(line))

    return inputs, outputs


questions, answers = load_conversations()

print('Sample question: {}'.format(questions[1]))
print('Sample answer: {}'.format(answers[1]))

# Build tokenizer using tfds for both questions and answers
tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    questions + answers, target_vocab_size=2 ** 13)

# Define start and end token to indicate the start and end of a sentence
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

# Vocabulary size plus start and end token
VOCAB_SIZE = tokenizer.vocab_size + 2

print('Tokenized sample question: {}'.format(tokenizer.encode(questions[20])))

# Maximum sentence length
MAX_LENGTH = 40


# Tokenize, filter and pad sentences
def tokenize_and_filter(inputs, outputs):
    tokenized_inputs, tokenized_outputs = [], []

    for (sentence1, sentence2) in zip(inputs, outputs):
        # tokenize sentence
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
        # check tokenized sentence max length
        if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)

    # pad tokenized sentences
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding='post')

    return tokenized_inputs, tokenized_outputs


questions, answers = tokenize_and_filter(questions, answers)

print('Vocab size: {}'.format(VOCAB_SIZE))
print('Number of samples: {}'.format(len(questions)))

decoder_target_data = []
for token_seq in answers:
    decoder_target_data.append(token_seq[1:])

answers = pad_sequences(decoder_target_data, maxlen=MAX_LENGTH, padding='post')
onehot_a_lines = utils.to_categorical(answers, VOCAB_SIZE)
decoder_target_data = np.array(onehot_a_lines)
print('Decoder target data shape -> {}'.format(decoder_target_data.shape))

# model
encoder_inputs = tf.keras.layers.Input(shape=( None , ))
encoder_embedding = tf.keras.layers.Embedding(VOCAB_SIZE, 256 , mask_zero=True ) (encoder_inputs)
encoder_outputs , state_h , state_c = tf.keras.layers.LSTM( 128 , return_state=True  )( encoder_embedding )
encoder_states = [ state_h , state_c ]

decoder_inputs = tf.keras.layers.Input(shape=( None ,  ))
decoder_embedding = tf.keras.layers.Embedding( VOCAB_SIZE, 256 , mask_zero=True) (decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM( 128 , return_state=True , return_sequences=True)
decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )
decoder_dense = tf.keras.layers.Dense( VOCAB_SIZE , activation=tf.keras.activations.softmax )
output = decoder_dense ( decoder_outputs )

model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output )
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy')

model.summary()

# TRAIN MODEL
model.fit([questions, answers], decoder_target_data, batch_size=250, epochs=10)
model.save('model.h5')


def evaluate(sentence):
    sentence = preprocess_sentence(sentence)

    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

    output = tf.expand_dims(START_TOKEN, 0)

    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        # concatenated the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(sentence):
    prediction = evaluate(sentence)

    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size])

    print('Input: {}'.format(sentence))
    print('Output: {}'.format(predicted_sentence))

    return predicted_sentence


sentence = 'I am not crazy, my mother had me tested, I promise'
for _ in range(5):
    sentence = predict(sentence)
    print('')
