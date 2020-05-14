import sys
import os

import Levenshtein

original_cwd = os.getcwd()
os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/nmt")
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/setup")
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/core")
from nmt import nmt
import argparse
from settings import hparams, hparams_name, out_dir, preprocessing, score as score_settings
sys.path.remove(os.path.dirname(os.path.realpath(__file__)) + "/setup")
import tensorflow as tf
from tokenizer import tokenize, detokenize, apply_bpe, apply_bpe_load
from sentence import replace_in_answers, normalize_new_lines
from scorer import score_answers
sys.path.remove(os.path.dirname(os.path.realpath(__file__)) + "/core")
import colorama
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from math import log, sqrt
import pandas as pd
import numpy as np
import re
import pickle

current_stdout = None

package_path = ''

classify_dir = os.path.join(package_path, "classify-models/")

key_profile_dir = os.path.join(package_path, "data-profile/retrieval_data/")


# That will not be as easy as training script, as code relies on input and output file in deep levels of code
# It also outputs massive amount of info
# We have to make own script for inference, so we could:cd ..
# - use it in interactive mode
# - import for use in other code
# - use input and output of our choice (so, for example, file as input and console as output,
#   or even console as input and file as output (but why? ;) ), etc)
# Why that nmt module doesn't give us some easy to use interface?

# Start inference "engine"
def do_start_inference(out_dir, hparams):

    # Silence all outputs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    global current_stdout

    current_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")

    # Modified autorun from nmt.py (bottom of the file)
    # We want to use original argument parser (for validation, etc)
    nmt_parser = argparse.ArgumentParser()
    nmt.add_arguments(nmt_parser)
    # But we have to hack settings from our config in there instead of commandline options
    flags, unparsed = nmt_parser.parse_known_args(['--'+k+'='+str(v) for k,v in hparams.items()])
    # And now we can run TF with modified arguments
    #tf.app.run(main=nmt.main, argv=[os.getcwd() + '\nmt\nmt\nmt.py'] + unparsed)
    # Add output (model) folder to flags
    flags.out_dir = out_dir

    # Make hparams
    hparams = nmt.create_hparams(flags)

    ## Train / Decode
    if not tf.gfile.Exists(flags.out_dir):
        nmt.utils.print_out("# Model folder (out_dir) doesn't exist")
        sys.exit()
    # Load hparams from model folder
    hparams = nmt.create_or_load_hparams(flags.out_dir, hparams, flags.hparams_path, save_hparams=False)
    # Choose checkpoint (provided with hparams or last one)
    if not flags.ckpt:
        flags.ckpt = tf.train.latest_checkpoint(flags.out_dir)

    # Create model
    model_creator = nmt.inference.get_model_creator(hparams)
    infer_model = nmt.inference.model_helper.create_infer_model(model_creator, hparams, None)
    sess, loaded_infer_model = nmt.inference.start_sess_and_load_model(infer_model, flags.ckpt)

    return (sess, infer_model, loaded_infer_model, flags, hparams)

# Inference
def do_inference(infer_data, sess, infer_model, loaded_infer_model, flags, hparams):

    # Disable TF logs for a while
    # Workaround for bug: https://github.com/tensorflow/tensorflow/issues/12414
    # Already fixed, available in nightly builds, but not in stable version
    # Maybe that will stay here to silence any outputs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    global current_stdout
    if not current_stdout:
        current_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    # With existing session
    with infer_model.graph.as_default():

        # Run model (translate)
        sess.run(
            infer_model.iterator.initializer,
            feed_dict={
                infer_model.src_placeholder: infer_data,
                infer_model.batch_size_placeholder: hparams.infer_batch_size
            })

        hparams.infer_mode = "beam_search"
        # calculate number of translations to be returned
        if hparams.infer_mode == "greedy" or "sample":
            num_translations_per_input = 1
        elif hparams.infer_mode == "beam_search":
            num_translations_per_input = min(hparams.num_translations_per_input, hparams.beam_width)



        answers = []
        while True:
            try:

                nmt_outputs, _ = loaded_infer_model.decode(sess)

                if hparams.infer_mode != "beam_search":
                    nmt_outputs = nmt.inference.nmt_model.np.expand_dims(nmt_outputs, 0)

                batch_size = nmt_outputs.shape[1]

                for sent_id in range(batch_size):

                    # Iterate through responses
                    translations = []
                    for beam_id in range(num_translations_per_input):

                        if hparams.eos:
                            tgt_eos = hparams.eos.encode("utf-8")

                        # Select a sentence
                        output = nmt_outputs[beam_id][sent_id, :].tolist()

                        # If there is an eos symbol in outputs, cut them at that point
                        if tgt_eos and tgt_eos in output:
                            output = output[:output.index(tgt_eos)]
                        print(output)

                        # Format response
                        if hparams.subword_option == "bpe":  # BPE
                            translation = nmt.utils.format_bpe_text(output)
                        elif hparams.subword_option == "spm":  # SPM
                            translation = nmt.utils.format_spm_text(output)
                        else:
                            translation = nmt.utils.format_text(output)

                        # Add response to the list
                        translations.append(translation.decode('utf-8'))

                    answers.append(translations)

            except tf.errors.OutOfRangeError:
                print("end")
                break

        # bug workaround end
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        sys.stdout.close()
        sys.stdout = current_stdout
        current_stdout = None

        return answers

# Fancy way to start everything on first inference() call
def start_inference(prepared_questions):

    global inference_helper, inference_object

    # Start inference, set global tuple with model, flags and hparams
    if False:
        inference_object = do_start_inference(os.path.join(package_path, "model-profile/"), hparams_name)
    else:
        inference_object = do_start_inference(out_dir, hparams)
    # First inference() call calls that method
    # Now we have everything running, so replace inference() with actual function call
    inference_helper = lambda question: do_inference(question, *inference_object)

    # Load BPE join pairs
    if preprocessing['use_bpe']:
        apply_bpe_load()

    # Rerun inference() call
    return inference_helper(prepared_questions)

# Model, flags and hparams
inference_object = None

# Function call helper (calls start_inference on first call, then do_inference)
inference_helper = start_inference

# Main inference function
def inference(questions, print = False):

    # Change current working directory (needed to load relative paths properly)
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Process questions
    answers_list = process_questions(questions)
    #answers = answers_list[0]

    # Revert current working directory
    os.chdir(original_cwd)

    # Return (one or more?)
    if not isinstance(questions, list):
        return answers_list[0]
    else:
        return answers_list

# Internal inference function (for direct call)
def inference_internal(questions):

    # Process questions and return
    return process_questions(questions, True)


# Get index and score for best answer
def get_best_score(answers_score):

    # Return first best scored response
    if score_settings['pick_random'] is None:
        max_score = max(answers_score)
        if max_score >= score_settings['bad_response_threshold']:
            return (answers_score.index(max_score), max_score)
        else:
            return (-1, None)

    # Return random best scored response
    elif score_settings['pick_random'] == 'best_score':
        indexes = [index for index, score in enumerate(answers_score) if score == max(answers_score) and score >= score_settings['bad_response_threshold']]
        if len(indexes):
            index = random.choice(indexes)
            return (index, answers_score[index])
        else:
            return (-1, None)

    # Return random response with score above threshold
    elif score_settings['pick_random'] == 'above_threshold':
        indexes = [index for index, score in enumerate(answers_score) if score > (score_settings['bad_response_threshold'] if score_settings['bad_response_threshold'] >= 0 else max(score)+score_settings['bad_response_threshold'])]
        if len(indexes):
            index = random.choice(indexes)
            return (index, answers_score[index])
        else:
            return (-1, None)

    return (0, score_settings['starting_score'])

# Process question or list of questions
def process_questions(questions, return_score_modifiers = False):

    # Make a list
    if not isinstance(questions, list):
        questions = [questions]

    # Clean and tokenize
    prepared_questions = []
    for question in questions:
        question = question.strip()
        prepared_questions.append(apply_bpe(tokenize(question)) if question else '##emptyquestion##')
    # Run inference
    answers_list = inference_helper(prepared_questions)

    # Process answers
    prepared_answers_list = []
    for index, answers in enumerate(answers_list):
        answers = detokenize(answers)
        answers = replace_in_answers(answers)
        answers = normalize_new_lines(answers)
        answers_score = score_answers(questions[index], answers)
        best_index, best_score = get_best_score(answers_score['score'])

        if prepared_questions[index] == '##emptyquestion##':
            prepared_answers_list.append(None)
        elif return_score_modifiers:
            prepared_answers_list.append({'answers': answers, 'scores': answers_score['score'], 'best_index': best_index, 'best_score': best_score, 'score_modifiers': answers_score['score_modifiers']})
        else:
            prepared_answers_list.append({'answers': answers, 'scores': answers_score['score'], 'best_index': best_index, 'best_score': best_score})
    return prepared_answers_list


def process_message(message, lower_case=True, stem=True, stop_words=True, gram=2):
    if lower_case:
        message = message.lower()
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i + gram])]
        return w
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    return words



def classify_question(question, models):
    predictions = []
    for model in models:
        if isinstance(question, list):
            sentence = model["tokenizer"].texts_to_sequences([question[0]])
        else:
            sentence = model["tokenizer"].texts_to_sequences([question])

        max_len = 36
        padded = pad_sequences(sentence, maxlen=max_len, padding='post')
        pred = model["model"].predict(padded)

        for item in pred:
            predictions.append({
                "profile": model["profile"],
                "score": float(item[0])
            })
    profile_prediction = {
        "profile": '',
        "score": 0
    }
    for prediction in predictions:
        if prediction["score"] > 0.77 and prediction["score"] > profile_prediction["score"]:
            profile_prediction["profile"] = prediction["profile"]
            profile_prediction["score"] = prediction["score"]

    if profile_prediction["profile"] == '':
        return 0
    else:
        print(question + ": " + str(profile_prediction["score"]) + str(profile_prediction["profile"]))
        return profile_prediction["profile"]


def retrieve_profile_answer(selected_profile, question, profile_value):
    with open(key_profile_dir + selected_profile + "_positive.from", "r") as from_file:
        with open(key_profile_dir + selected_profile + "_positive_to_with_placeholder", "r") as to_file:
            question_set = []
            answer_set = []
            q_lines = from_file.readlines()
            a_lines = to_file.readlines()
            for line in q_lines:
                question_set.append(line)
            for line in a_lines:
                answer_set.append(line)

        ratios = []
        i = 0
        while i < len(question_set):
            ratio = Levenshtein.ratio(question, question_set[i])
            ratios.append({
                "question": question_set[i],
                "ratio": float(ratio),
                "answer": answer_set[i]
            })
            i += 1

        best_ratio = {
            "question": '',
            "answer": '',
            "ratio": 0
        }
        for prediction in ratios:
            if  prediction["ratio"] > best_ratio["ratio"]:
                best_ratio["question"] = prediction["question"]
                best_ratio["answer"] = prediction["answer"]
                best_ratio["ratio"] = prediction["ratio"]

        print(best_ratio["answer"].replace("<" + selected_profile + ">", profile_value))
        return best_ratio["answer"].replace("<" + selected_profile + ">", profile_value)


# interactive mode
if __name__ == "__main__":

    # Input file
    if sys.stdin.isatty() == False:

        # Process questions
        answers_list = process_questions(sys.stdin.readlines())

        # Print answers
        for answers in answers_list:
            print(answers['answers'][answers['best_index']])

        sys.exit()

    # Interactive mode
    colorama.init()
    print("\n\nStarting interactive mode (first response will take a while):")

    # Specified model
    if len(sys.argv) >= 2 and sys.argv[1]:
        checkpoint = hparams['out_dir'] + str(sys.argv[1])
        hparams['ckpt'] = checkpoint
        print("Using checkpoint: {}".format(checkpoint))

    profiles = ['name', 'age', 'gender', 'location', 'hobby']
    models = []
    for profile in profiles:
        with open(classify_dir + 'tokenizer_' + profile + '.pickle', 'rb') as handle:
            loaded_tokenizer = pickle.load(handle)
        loaded_model = tf.keras.models.load_model(classify_dir + profile +'_classify_model.h5')
        models.append({
            "model": loaded_model,
            "tokenizer": loaded_tokenizer,
            "profile": profile

        })

    # QAs
    profile_name = input("Please enter my name: ")
    profile_age = input("Please enter my age: ")
    profile_gender = input("Please enter my gender: ")
    profile_location = input("Please enter my location: ")
    profile_hobby = input("Please enter my favourite hobby: ")
    profile_vector = {
        "name": profile_name,
        "age": profile_age,
        "gender": profile_gender,
        "location": profile_location,
        "hobby": profile_hobby,
    }
    while True:
        question = input("\n> ")
        if classify_question(question, models) is not 0:
            selected_profile = classify_question(question,models)
            retrieve_profile_answer(selected_profile, question, profile_vector[selected_profile])
        else:
            answers = inference_internal(question)[0]
            if answers is None:
                print(colorama.Fore.RED + "! Question can't be empty" + colorama.Fore.RESET)
            else:
                for i, _ in enumerate(answers['scores']):
                    if answers['scores'][i] == max(answers['scores']):
                        print(answers['answers'][i])
                    print(answers['answers'])
                    print("{}- {}{} [{}] {}{}{}".format(colorama.Fore.GREEN if answers['scores'][i] == max(answers['scores']) and answers['scores'][i] >= score_settings['bad_response_threshold'] else colorama.Fore.YELLOW if answers['scores'][i] >= score_settings['bad_response_threshold'] else colorama.Fore.RED, answers['answers'][i], colorama.Fore.RESET, answers['scores'][i], colorama.Fore.BLUE, answers['score_modifiers'][i] if score_settings['show_score_modifiers'] else '', colorama.Fore.RESET))

os.chdir(original_cwd)
