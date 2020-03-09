import json
import nltk
import json
import numpy as np
import tensorflow as tf
import tflearn
import pickle
import random
from nltk.stem.lancaster import LancasterStemmer
from glob import glob
from sentiment import analyze_text

stemmer = LancasterStemmer()

words = []
labels = []
docs_x = []
docs_y = []
datasets = ['video_games']
new_intents = []

for set in datasets:
    qs = []
    ans = []
    for file_name in glob("amazon_qa/" + set + "/*.json"):
        for line in open(file_name):
            file = json.loads(line)
            question = [file['context']]
            answer = [file['response']]
            # print(questions)
            qs.extend(question)
            ans.extend(answer)
            # print([file['context']])

        new_intent = {
                      "tag": set,
                      "patterns": qs,
                      "responses": ans,
                      "context_set": ""
                      }

    new_intents.append(new_intent.copy())

with open('result.json', 'w') as fp:
    json.dump(new_intents, fp)

#  for q in qs:
#      print(q)

# for a in ans:
#     print(a)
#   wrds = nltk.word_tokenize(question)
#  words.extend(wrds)
# docs_x.append(wrds)

# questions.extend(file('context'))
