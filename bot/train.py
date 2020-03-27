from __future__ import absolute_import, division, print_function, unicode_literals
from keras.preprocessing import sequence
import tensorflow_datasets as tfds
from tensorflow import TensorShape
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.layers import Input, GRU, Dense, Concatenate, TimeDistributed, Embedding, LSTM
from keras.activations import softmax
from keras.optimizers import RMSprop, Adam
from keras.models import Model
from keras import layers, activations, models, preprocessing, utils
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
import re
import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt


# for seq_index in range(20):
#     # Take one sequence (part of the training test)
#     # for trying out decoding.
#     input_seq = original[seq_index: seq_index + 1]
#     decoded_sentence = decode_sequence(input_seq)
#     print('-')
#     print('Decoded sentence:', decoded_sentence)


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
    with open("test.from", 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        inputs.append(clean_line(line))

    with open("test.to", 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        outputs.append(clean_line(line))

    return inputs, outputs


questions, answers = load_conversations()

print('Sample question: {}'.format(questions[1]))
print('Sample answer: {}'.format(answers[1]))

tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    questions + answers, target_vocab_size=2**13)
# Define start and end token to indicate the start and end of a sentence
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

# Vocabulary size plus start and end token
VOCAB_SIZE = tokenizer.vocab_size + 2

# Maximum sentence length
MAX_LENGTH = 40


# Tokenize, filter and pad sentences
# tokenizer.fit_on_texts(questions)
# tokenizer.fit_on_texts(answers)
#tokenized_questions, tokenized_answers = tokenizer.texts_to_sequences(questions, answers)

# length_list = list()
# for token_seq in tokenized_questions:
#     length_list.append(len(token_seq))
# max_input_length = np.array(length_list).max()
# print('Question max length is {}'.format(max_input_length))
#
# padded_questions = preprocessing.sequence.pad_sequences(tokenized_questions, maxlen=max_input_length, padding='post')
# encoder_input_data = np.array(padded_questions)
# print('Encoder input data shape -> {}'.format(encoder_input_data.shape))
#
# q_dict = tokenizer.word_index
# num_q_tokens = len(q_dict)+1
# print('Number of question tokens = {}'.format(num_q_tokens))
#
# for token_seq in tokenized_answers:
#     length_list.append( len( token_seq ))
# max_output_length = np.array( length_list ).max()
# print('Answers max length is {}'.format( max_output_length ))
#
# padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=max_output_length, padding='post' )
# decoder_input_data = np.array(padded_answers)
# print('Decoder input data shape -> {}'.format(decoder_input_data.shape))
#
# a_dict = tokenizer.word_index
# num_a_tokens = len(a_dict)+1
# print('Number of answer tokens = {}'.format(num_a_tokens))
#
# print('Tokenized sample question: {}'.format(tokenizer.encode(questions[20])))
#
#
# # Tokenize, filter and pad sentences
#
#
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
    tokenized_inputs = sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
    print("num input token = {}".format(tokenized_inputs))
    tokenized_outputs = sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding='post')
    print("num output token = {}".format(tokenized_outputs))
    return tokenized_inputs, tokenized_outputs


questions, answers = tokenize_and_filter(questions, answers)
#
print('Vocab size: {}'.format(VOCAB_SIZE))
print('Number of samples: {}'.format(len(questions)))
#
decoder_target_data = list()
for token_seq in answers:
    decoder_target_data.append(token_seq[1:])

answers = pad_sequences(decoder_target_data, maxlen=MAX_LENGTH, padding='post')
onehot_a_lines = utils.to_categorical(answers, VOCAB_SIZE)
decoder_target_data = np.array(onehot_a_lines)
print('Decoder target data shape -> {}'.format(decoder_target_data.shape))

# model
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(VOCAB_SIZE, 256, mask_zero=True)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(128, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(VOCAB_SIZE, 256, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(128, return_state=True, return_sequences=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(VOCAB_SIZE, activation=softmax)
output = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], output)
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy')

model.summary()


# TRAIN MODEL
# model.fit([questions, answers], decoder_target_data, batch_size=250, epochs=10)
# model.save('model.h5')


# for seq_index in range(20):
#     # Take one sequence (part of the training test)
#     # for trying out decoding.
#     input_seq = original[seq_index: seq_index + 1]
#     decoded_sentence = decode_sequence(input_seq)
#     print('-')
#     print('Decoded sentence:', decoded_sentence)

#
# questions, answers = load_conversations()
#
# print('Sample question: {}'.format(questions[1]))
# print('Sample answer: {}'.format(answers[1]))

# # Build tokenizer using tfds for both questions and answers
# tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
#     questions + answers, target_vocab_size=2 ** 13)
#
# # Define start and end token to indicate the start and end of a sentence
# START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
#
# # Vocabulary size plus start and end token
# VOCAB_SIZE = tokenizer.vocab_size + 2
#
# # Maximum sentence length
# MAX_LENGTH = 40
#
#
# tokenizer = preprocessing.text.Tokenizer()
#
# tokenizer.fit_on_texts(questions)
# tokenizer.fit_on_texts(answers)
# tokenized_questions, tokenized_answers = tokenizer.texts_to_sequences(questions, answers)
#
# length_list = list()
# for token_seq in tokenized_questions:
#     length_list.append(len(token_seq))
# max_input_length = np.array(length_list).max()
# print('Question max length is {}'.format(max_input_length))
#
# padded_questions = preprocessing.sequence.pad_sequences(tokenized_questions, maxlen=max_input_length, padding='post')
# encoder_input_data = np.array(padded_questions)
# print('Encoder input data shape -> {}'.format(encoder_input_data.shape))
#
# q_dict = tokenizer.word_index
# num_q_tokens = len(q_dict)+1
# print('Number of question tokens = {}'.format(num_q_tokens))
#
# for token_seq in tokenized_answers:
#     length_list.append( len( token_seq ))
# max_output_length = np.array( length_list ).max()
# print('Answers max length is {}'.format( max_output_length ))
#
# padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=max_output_length, padding='post' )
# decoder_input_data = np.array(padded_answers)
# print('Decoder input data shape -> {}'.format(decoder_input_data.shape))
#
# a_dict = tokenizer.word_index
# num_a_tokens = len(a_dict)+1
# print('Number of answer tokens = {}'.format(num_a_tokens))


#   TRAIN MODEL
# model.fit([questions, answers], decoder_target_data, batch_size=250, epochs=10)
# model.save('model.h5')

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


output = predict('hello')
# def make_inference_models():
#     encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)
#
#     decoder_state_input_h = tf.keras.layers.Input(shape=(128,))
#     decoder_state_input_c = tf.keras.layers.Input(shape=(128,))
#
#     decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
#
#     decoder_outputs, state_h, state_c = decoder_lstm(
#         decoder_embedding, initial_state=decoder_states_inputs)
#     decoder_states = [state_h, state_c]
#     decoder_outputs = decoder_dense(decoder_outputs)
#     decoder_model = tf.keras.models.Model(
#         [decoder_inputs] + decoder_states_inputs,
#         [decoder_outputs] + decoder_states)
#
#     return encoder_model, decoder_model
#
#
# def str_to_tokens(sentence: str):
#     words = sentence.lower().split()
#     tokens_list = list()
#     for word in words:
#         tokens_list.append(eng_word_dict[word])
#     return preprocessing.sequence.pad_sequences([tokens_list], maxlen=MAX_LENGTH, padding='post')
#
#
# enc_model, dec_model = make_inference_models()
#
# for epoch in range(questions.shape[0]):
#     states_values = enc_model.predict(str_to_tokens(input('Enter eng sentence : ')))
#     # states_values = enc_model.predict( encoder_input_data[ epoch ] )
#     empty_target_seq = np.zeros((1, 1))
#     empty_target_seq[0, 0] = mar_word_dict['start']
#     stop_condition = False
#     decoded_translation = ''
#     while not stop_condition:
#         dec_outputs, h, c = dec_model.predict([empty_target_seq] + states_values)
#         sampled_word_index = np.argmax(dec_outputs[0, -1, :])
#         sampled_word = None
#         for word, index in mar_word_dict.items():
#             if sampled_word_index == index:
#                 decoded_translation += ' {}'.format(word)
#                 sampled_word = word
#
#         if sampled_word == 'end' or len(decoded_translation.split()) > max_output_length:
#             stop_condition = True
#
#         empty_target_seq = np.zeros((1, 1))
#         empty_target_seq[0, 0] = sampled_word_index
#         states_values = [h, c]
#
#     print(decoded_translation)


#
# BATCH_SIZE = 64
# BUFFER_SIZE = 20000
#
# # decoder inputs use the previous target as input
# # remove START_TOKEN from targets
# dataset = tf.data.Dataset.from_tensor_slices((
#     {
#         'inputs': questions,
#         'dec_inputs': answers[:, :-1]
#     },
#     {
#         'outputs': answers[:, 1:]
#     },
# ))
#
# dataset = dataset.cache()
# dataset = dataset.shuffle(BUFFER_SIZE)
# dataset = dataset.batch(BATCH_SIZE)
# dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
#
# print(dataset)
#
#
# def scaled_dot_product_attention(query, key, value, mask):
#     """Calculate the attention weights. """
#     matmul_qk = tf.matmul(query, key, transpose_b=True)
#
#     # scale matmul_qk
#     depth = tf.cast(tf.shape(key)[-1], tf.float32)
#     logits = matmul_qk / tf.math.sqrt(depth)
#
#     # add the mask to zero out padding tokens
#     if mask is not None:
#         logits += (mask * -1e9)
#
#     # softmax is normalized on the last axis (seq_len_k)
#     attention_weights = tf.nn.softmax(logits, axis=-1)
#
#     output = tf.matmul(attention_weights, value)
#
#     return output
#
#
# class MultiHeadAttention(tf.keras.layers.Layer):
#
#     def __init__(self, d_model, num_heads, name="multi_head_attention"):
#         super(MultiHeadAttention, self).__init__(name=name)
#         self.num_heads = num_heads
#         self.d_model = d_model
#
#         assert d_model % self.num_heads == 0
#
#         self.depth = d_model // self.num_heads
#
#         self.query_dense = tf.keras.layers.Dense(units=d_model)
#         self.key_dense = tf.keras.layers.Dense(units=d_model)
#         self.value_dense = tf.keras.layers.Dense(units=d_model)
#
#         self.dense = tf.keras.layers.Dense(units=d_model)
#
#     def split_heads(self, inputs, batch_size):
#         inputs = tf.reshape(
#             inputs, shape=(batch_size, -1, self.num_heads, self.depth))
#         return tf.transpose(inputs, perm=[0, 2, 1, 3])
#
#     def call(self, inputs):
#         query, key, value, mask = inputs['query'], inputs['key'], inputs[
#             'value'], inputs['mask']
#         batch_size = tf.shape(query)[0]
#
#         # linear layers
#         query = self.query_dense(query)
#         key = self.key_dense(key)
#         value = self.value_dense(value)
#
#         # split heads
#         query = self.split_heads(query, batch_size)
#         key = self.split_heads(key, batch_size)
#         value = self.split_heads(value, batch_size)
#
#         # scaled dot-product attention
#         scaled_attention = scaled_dot_product_attention(query, key, value, mask)
#
#         scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
#
#         # concatenation of heads
#         concat_attention = tf.reshape(scaled_attention,
#                                       (batch_size, -1, self.d_model))
#
#         # final linear layer
#         outputs = self.dense(concat_attention)
#
#         return outputs
#
#
# def create_padding_mask(x):
#     mask = tf.cast(tf.math.equal(x, 0), tf.float32)
#     # (batch_size, 1, 1, sequence length)
#     return mask[:, tf.newaxis, tf.newaxis, :]
#
#
# def create_look_ahead_mask(x):
#     seq_len = tf.shape(x)[1]
#     look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
#     padding_mask = create_padding_mask(x)
#     return tf.maximum(look_ahead_mask, padding_mask)
#
#
# class PositionalEncoding(tf.keras.layers.Layer):
#
#     def __init__(self, position, d_model):
#         super(PositionalEncoding, self).__init__()
#         self.pos_encoding = self.positional_encoding(position, d_model)
#
#     def get_angles(self, position, i, d_model):
#         angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
#         return position * angles
#
#     def positional_encoding(self, position, d_model):
#         angle_rads = self.get_angles(
#             position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
#             i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
#             d_model=d_model)
#         # apply sin to even index in the array
#         sines = tf.math.sin(angle_rads[:, 0::2])
#         # apply cos to odd index in the array
#         cosines = tf.math.cos(angle_rads[:, 1::2])
#
#         pos_encoding = tf.concat([sines, cosines], axis=-1)
#         pos_encoding = pos_encoding[tf.newaxis, ...]
#         return tf.cast(pos_encoding, tf.float32)
#
#     def call(self, inputs):
#         return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
#
#
# sample_pos_encoding = PositionalEncoding(50, 512)
#
# plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
# plt.xlabel('Depth')
# plt.xlim((0, 512))
# plt.ylabel('Position')
# plt.colorbar()
# plt.show()
#
#
# def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
#     inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
#     padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
#
#     attention = MultiHeadAttention(
#         d_model, num_heads, name="attention")({
#         'query': inputs,
#         'key': inputs,
#         'value': inputs,
#         'mask': padding_mask
#     })
#     attention = tf.keras.layers.Dropout(rate=dropout)(attention)
#     attention = tf.keras.layers.LayerNormalization(
#         epsilon=1e-6)(inputs + attention)
#
#     outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
#     outputs = tf.keras.layers.Dense(units=d_model)(outputs)
#     outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
#     outputs = tf.keras.layers.LayerNormalization(
#         epsilon=1e-6)(attention + outputs)
#
#     return tf.keras.Model(
#         inputs=[inputs, padding_mask], outputs=outputs, name=name)
#
#
# def encoder(vocab_size,
#             num_layers,
#             units,
#             d_model,
#             num_heads,
#             dropout,
#             name="encoder"):
#     inputs = tf.keras.Input(shape=(None,), name="inputs")
#     padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
#
#     embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
#     embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
#     embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
#
#     outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)
#
#     for i in range(num_layers):
#         outputs = encoder_layer(
#             units=units,
#             d_model=d_model,
#             num_heads=num_heads,
#             dropout=dropout,
#             name="encoder_layer_{}".format(i),
#         )([outputs, padding_mask])
#
#     return tf.keras.Model(
#         inputs=[inputs, padding_mask], outputs=outputs, name=name)
#
#
# def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
#     inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
#     enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
#     look_ahead_mask = tf.keras.Input(
#         shape=(1, None, None), name="look_ahead_mask")
#     padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
#
#     attention1 = MultiHeadAttention(
#         d_model, num_heads, name="attention_1")(inputs={
#         'query': inputs,
#         'key': inputs,
#         'value': inputs,
#         'mask': look_ahead_mask
#     })
#     attention1 = tf.keras.layers.LayerNormalization(
#         epsilon=1e-6)(attention1 + inputs)
#
#     attention2 = MultiHeadAttention(
#         d_model, num_heads, name="attention_2")(inputs={
#         'query': attention1,
#         'key': enc_outputs,
#         'value': enc_outputs,
#         'mask': padding_mask
#     })
#     attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
#     attention2 = tf.keras.layers.LayerNormalization(
#         epsilon=1e-6)(attention2 + attention1)
#
#     outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
#     outputs = tf.keras.layers.Dense(units=d_model)(outputs)
#     outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
#     outputs = tf.keras.layers.LayerNormalization(
#         epsilon=1e-6)(outputs + attention2)
#
#     return tf.keras.Model(
#         inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
#         outputs=outputs,
#         name=name)
#
#
# def decoder(vocab_size,
#             num_layers,
#             units,
#             d_model,
#             num_heads,
#             dropout,
#             name='decoder'):
#     inputs = tf.keras.Input(shape=(None,), name='inputs')
#     enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
#     look_ahead_mask = tf.keras.Input(
#         shape=(1, None, None), name='look_ahead_mask')
#     padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
#
#     embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
#     embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
#     embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
#
#     outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)
#
#     for i in range(num_layers):
#         outputs = decoder_layer(
#             units=units,
#             d_model=d_model,
#             num_heads=num_heads,
#             dropout=dropout,
#             name='decoder_layer_{}'.format(i),
#         )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])
#
#     return tf.keras.Model(
#         inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
#         outputs=outputs,
#         name=name)
#
#
# def transformer(vocab_size,
#                 num_layers,
#                 units,
#                 d_model,
#                 num_heads,
#                 dropout,
#                 name="transformer"):
#     inputs = tf.keras.Input(shape=(None,), name="inputs")
#     dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")
#
#     enc_padding_mask = tf.keras.layers.Lambda(
#         create_padding_mask, output_shape=(1, 1, None),
#         name='enc_padding_mask')(inputs)
#     # mask the future tokens for decoder inputs at the 1st attention block
#     look_ahead_mask = tf.keras.layers.Lambda(
#         create_look_ahead_mask,
#         output_shape=(1, None, None),
#         name='look_ahead_mask')(dec_inputs)
#     # mask the encoder outputs for the 2nd attention block
#     dec_padding_mask = tf.keras.layers.Lambda(
#         create_padding_mask, output_shape=(1, 1, None),
#         name='dec_padding_mask')(inputs)
#
#     enc_outputs = encoder(
#         vocab_size=vocab_size,
#         num_layers=num_layers,
#         units=units,
#         d_model=d_model,
#         num_heads=num_heads,
#         dropout=dropout,
#     )(inputs=[inputs, enc_padding_mask])
#
#     dec_outputs = decoder(
#         vocab_size=vocab_size,
#         num_layers=num_layers,
#         units=units,
#         d_model=d_model,
#         num_heads=num_heads,
#         dropout=dropout,
#     )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])
#
#     outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)
#
#     return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)
#
#
# tf.keras.backend.clear_session()
#
# # Hyper-parameters
# NUM_LAYERS = 2
# D_MODEL = 256
# NUM_HEADS = 8
# UNITS = 512
# DROPOUT = 0.1
#
# model = transformer(
#     vocab_size=VOCAB_SIZE,
#     num_layers=NUM_LAYERS,
#     units=UNITS,
#     d_model=D_MODEL,
#     num_heads=NUM_HEADS,
#     dropout=DROPOUT)
#
#
# def loss_function(y_true, y_pred):
#     y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
#
#     loss = tf.keras.losses.SparseCategoricalCrossentropy(
#         from_logits=True, reduction='none')(y_true, y_pred)
#
#     mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
#     loss = tf.multiply(loss, mask)
#
#     return tf.reduce_mean(loss)
#
#
# optimizer = tf.keras.optimizers.Adam(
#     lr=0.001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
#
#
# def accuracy(y_true, y_pred):
#     # ensure labels have shape (batch_size, MAX_LENGTH - 1)
#     y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
#     return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
#
#
# model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])
#
# EPOCHS = 20
#
# model.fit(dataset, epochs=EPOCHS)
#


# def make_inference_models():
#     encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)
#
#     decoder_state_input_h = tf.keras.layers.Input(shape=(128,))
#     decoder_state_input_c = tf.keras.layers.Input(shape=(128,))
#
#     decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
#
#     decoder_outputs, state_h, state_c = decoder_lstm(
#         decoder_embedding, initial_state=decoder_states_inputs)
#     decoder_states = [state_h, state_c]
#     decoder_outputs = decoder_dense(decoder_outputs)
#     decoder_model = tf.keras.models.Model(
#         [decoder_inputs] + decoder_states_inputs,
#         [decoder_outputs] + decoder_states)
#
#     return encoder_model, decoder_model
#
#
# def str_to_tokens(sentence: str):
#     words = sentence.lower().split()
#     tokens_list = list()
#     for word in words:
#         tokens_list.append(eng_word_dict[word])
#     return preprocessing.sequence.pad_sequences([tokens_list], maxlen=MAX_LENGTH, padding='post')
#
#
# enc_model, dec_model = make_inference_models()
#
# for epoch in range(questions.shape[0]):
#     states_values = enc_model.predict(str_to_tokens(input('Enter eng sentence : ')))
#     # states_values = enc_model.predict( encoder_input_data[ epoch ] )
#     empty_target_seq = np.zeros((1, 1))
#     empty_target_seq[0, 0] = mar_word_dict['start']
#     stop_condition = False
#     decoded_translation = ''
#     while not stop_condition:
#         dec_outputs, h, c = dec_model.predict([empty_target_seq] + states_values)
#         sampled_word_index = np.argmax(dec_outputs[0, -1, :])
#         sampled_word = None
#         for word, index in mar_word_dict.items():
#             if sampled_word_index == index:
#                 decoded_translation += ' {}'.format(word)
#                 sampled_word = word
#
#         if sampled_word == 'end' or len(decoded_translation.split()) > max_output_length:
#             stop_condition = True
#
#         empty_target_seq = np.zeros((1, 1))
#         empty_target_seq[0, 0] = sampled_word_index
#         states_values = [h, c]
#
#     print(decoded_translation)


#
# BATCH_SIZE = 64
# BUFFER_SIZE = 20000
#
# # decoder inputs use the previous target as input
# # remove START_TOKEN from targets
# dataset = tf.data.Dataset.from_tensor_slices((
#     {
#         'inputs': questions,
#         'dec_inputs': answers[:, :-1]
#     },
#     {
#         'outputs': answers[:, 1:]
#     },
# ))
#
# dataset = dataset.cache()
# dataset = dataset.shuffle(BUFFER_SIZE)
# dataset = dataset.batch(BATCH_SIZE)
# dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
#
# print(dataset)
#
#
# def scaled_dot_product_attention(query, key, value, mask):
#     """Calculate the attention weights. """
#     matmul_qk = tf.matmul(query, key, transpose_b=True)
#
#     # scale matmul_qk
#     depth = tf.cast(tf.shape(key)[-1], tf.float32)
#     logits = matmul_qk / tf.math.sqrt(depth)
#
#     # add the mask to zero out padding tokens
#     if mask is not None:
#         logits += (mask * -1e9)
#
#     # softmax is normalized on the last axis (seq_len_k)
#     attention_weights = tf.nn.softmax(logits, axis=-1)
#
#     output = tf.matmul(attention_weights, value)
#
#     return output
#
#
# class MultiHeadAttention(tf.keras.layers.Layer):
#
#     def __init__(self, d_model, num_heads, name="multi_head_attention"):
#         super(MultiHeadAttention, self).__init__(name=name)
#         self.num_heads = num_heads
#         self.d_model = d_model
#
#         assert d_model % self.num_heads == 0
#
#         self.depth = d_model // self.num_heads
#
#         self.query_dense = tf.keras.layers.Dense(units=d_model)
#         self.key_dense = tf.keras.layers.Dense(units=d_model)
#         self.value_dense = tf.keras.layers.Dense(units=d_model)
#
#         self.dense = tf.keras.layers.Dense(units=d_model)
#
#     def split_heads(self, inputs, batch_size):
#         inputs = tf.reshape(
#             inputs, shape=(batch_size, -1, self.num_heads, self.depth))
#         return tf.transpose(inputs, perm=[0, 2, 1, 3])
#
#     def call(self, inputs):
#         query, key, value, mask = inputs['query'], inputs['key'], inputs[
#             'value'], inputs['mask']
#         batch_size = tf.shape(query)[0]
#
#         # linear layers
#         query = self.query_dense(query)
#         key = self.key_dense(key)
#         value = self.value_dense(value)
#
#         # split heads
#         query = self.split_heads(query, batch_size)
#         key = self.split_heads(key, batch_size)
#         value = self.split_heads(value, batch_size)
#
#         # scaled dot-product attention
#         scaled_attention = scaled_dot_product_attention(query,  key, value, mask)
#
#         scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
#
#         # concatenation of heads
#         concat_attention = tf.reshape(scaled_attention,
#                                       (batch_size, -1, self.d_model))
#
#         # final linear layer
#         outputs = self.dense(concat_attention)
#
#         return outputs
#
#
# def create_padding_mask(x):
#     mask = tf.cast(tf.math.equal(x, 0), tf.float32)
#     # (batch_size, 1, 1, sequence length)
#     return mask[:, tf.newaxis, tf.newaxis, :]
#
#
# def create_look_ahead_mask(x):
#     seq_len = tf.shape(x)[1]
#     look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
#     padding_mask = create_padding_mask(x)
#     return tf.maximum(look_ahead_mask, padding_mask)
#
#
# class PositionalEncoding(tf.keras.layers.Layer):
#
#     def __init__(self, position, d_model):
#         super(PositionalEncoding, self).__init__()
#         self.pos_encoding = self.positional_encoding(position, d_model)
#
#     def get_angles(self, position, i, d_model):
#         angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
#         return position * angles
#
#     def positional_encoding(self, position, d_model):
#         angle_rads = self.get_angles(
#             position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
#             i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
#             d_model=d_model)
#         # apply sin to even index in the array
#         sines = tf.math.sin(angle_rads[:, 0::2])
#         # apply cos to odd index in the array
#         cosines = tf.math.cos(angle_rads[:, 1::2])
#
#         pos_encoding = tf.concat([sines, cosines], axis=-1)
#         pos_encoding = pos_encoding[tf.newaxis, ...]
#         return tf.cast(pos_encoding, tf.float32)
#
#     def call(self, inputs):
#         return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
#
#
# sample_pos_encoding = PositionalEncoding(50, 512)
#
# plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
# plt.xlabel('Depth')
# plt.xlim((0, 512))
# plt.ylabel('Position')
# plt.colorbar()
# plt.show()
#
#
# def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
#     inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
#     padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
#
#     attention = MultiHeadAttention(
#         d_model, num_heads, name="attention")({
#         'query': inputs,
#         'key': inputs,
#         'value': inputs,
#         'mask': padding_mask
#     })
#     attention = tf.keras.layers.Dropout(rate=dropout)(attention)
#     attention = tf.keras.layers.LayerNormalization(
#         epsilon=1e-6)(inputs + attention)
#
#     outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
#     outputs = tf.keras.layers.Dense(units=d_model)(outputs)
#     outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
#     outputs = tf.keras.layers.LayerNormalization(
#         epsilon=1e-6)(attention + outputs)
#
#     return tf.keras.Model(
#         inputs=[inputs, padding_mask], outputs=outputs, name=name)
#
#
# def encoder(vocab_size,
#             num_layers,
#             units,
#             d_model,
#             num_heads,
#             dropout,
#             name="encoder"):
#     inputs = tf.keras.Input(shape=(None,), name="inputs")
#     padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
#
#     embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
#     embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
#     embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
#
#     outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)
#
#     for i in range(num_layers):
#         outputs = encoder_layer(
#             units=units,
#             d_model=d_model,
#             num_heads=num_heads,
#             dropout=dropout,
#             name="encoder_layer_{}".format(i),
#         )([outputs, padding_mask])
#
#     return tf.keras.Model(
#         inputs=[inputs, padding_mask], outputs=outputs, name=name)
#
#
# def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
#     inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
#     enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
#     look_ahead_mask = tf.keras.Input(
#         shape=(1, None, None), name="look_ahead_mask")
#     padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
#
#     attention1 = MultiHeadAttention(
#         d_model, num_heads, name="attention_1")(inputs={
#         'query': inputs,
#         'key': inputs,
#         'value': inputs,
#         'mask': look_ahead_mask
#     })
#     attention1 = tf.keras.layers.LayerNormalization(
#         epsilon=1e-6)(attention1 + inputs)
#
#     attention2 = MultiHeadAttention(
#         d_model, num_heads, name="attention_2")(inputs={
#         'query': attention1,
#         'key': enc_outputs,
#         'value': enc_outputs,
#         'mask': padding_mask
#     })
#     attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
#     attention2 = tf.keras.layers.LayerNormalization(
#         epsilon=1e-6)(attention2 + attention1)
#
#     outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
#     outputs = tf.keras.layers.Dense(units=d_model)(outputs)
#     outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
#     outputs = tf.keras.layers.LayerNormalization(
#         epsilon=1e-6)(outputs + attention2)
#
#     return tf.keras.Model(
#         inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
#         outputs=outputs,
#         name=name)
#
#
# def decoder(vocab_size,
#             num_layers,
#             units,
#             d_model,
#             num_heads,
#             dropout,
#             name='decoder'):
#     inputs = tf.keras.Input(shape=(None,), name='inputs')
#     enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
#     look_ahead_mask = tf.keras.Input(
#         shape=(1, None, None), name='look_ahead_mask')
#     padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
#
#     embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
#     embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
#     embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
#
#     outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)
#
#     for i in range(num_layers):
#         outputs = decoder_layer(
#             units=units,
#             d_model=d_model,
#             num_heads=num_heads,
#             dropout=dropout,
#             name='decoder_layer_{}'.format(i),
#         )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])
#
#     return tf.keras.Model(
#         inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
#         outputs=outputs,
#         name=name)
#
#
# def transformer(vocab_size,
#                 num_layers,
#                 units,
#                 d_model,
#                 num_heads,
#                 dropout,
#                 name="transformer"):
#     inputs = tf.keras.Input(shape=(None,), name="inputs")
#     dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")
#
#     enc_padding_mask = tf.keras.layers.Lambda(
#         create_padding_mask, output_shape=(1, 1, None),
#         name='enc_padding_mask')(inputs)
#     # mask the future tokens for decoder inputs at the 1st attention block
#     look_ahead_mask = tf.keras.layers.Lambda(
#         create_look_ahead_mask,
#         output_shape=(1, None, None),
#         name='look_ahead_mask')(dec_inputs)
#     # mask the encoder outputs for the 2nd attention block
#     dec_padding_mask = tf.keras.layers.Lambda(
#         create_padding_mask, output_shape=(1, 1, None),
#         name='dec_padding_mask')(inputs)
#
#     enc_outputs = encoder(
#         vocab_size=vocab_size,
#         num_layers=num_layers,
#         units=units,
#         d_model=d_model,
#         num_heads=num_heads,
#         dropout=dropout,
#     )(inputs=[inputs, enc_padding_mask])
#
#     dec_outputs = decoder(
#         vocab_size=vocab_size,
#         num_layers=num_layers,
#         units=units,
#         d_model=d_model,
#         num_heads=num_heads,
#         dropout=dropout,
#     )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])
#
#     outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)
#
#     return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)
#
#
# tf.keras.backend.clear_session()
#
# # Hyper-parameters
# NUM_LAYERS = 2
# D_MODEL = 256
# NUM_HEADS = 8
# UNITS = 512
# DROPOUT = 0.1
#
# model = transformer(
#     vocab_size=VOCAB_SIZE,
#     num_layers=NUM_LAYERS,
#     units=UNITS,
#     d_model=D_MODEL,
#     num_heads=NUM_HEADS,
#     dropout=DROPOUT)
#
#
# def loss_function(y_true, y_pred):
#     y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
#
#     loss = tf.keras.losses.SparseCategoricalCrossentropy(
#         from_logits=True, reduction='none')(y_true, y_pred)
#
#     mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
#     loss = tf.multiply(loss, mask)
#
#     return tf.reduce_mean(loss)
#
#
# optimizer = tf.keras.optimizers.Adam(
#     lr=0.001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
#
#
# def accuracy(y_true, y_pred):
#     # ensure labels have shape (batch_size, MAX_LENGTH - 1)
#     y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
#     return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
#
#
# model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])
#
# EPOCHS = 20
#
# model.fit(dataset, epochs=EPOCHS)
#
#

