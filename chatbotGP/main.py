# -*- coding: utf-8 -*-
# encoding: utf-8
import nltk

nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
#from nltk.stem.isri import ISRIStemmer

stemmer = LancasterStemmer()

import numpy
import tensorflow
import tflearn
import random
import json
import pickle
from ftfy import fix_encoding
import re
import sys

with open("intents.json",'r', encoding="utf-8") as file:
    data = json.load(file)
#print(data)

#try:
 #   with open("data.pickle", "rb") as f:
  #      words, labels, training, output = pickle.load(f)
#except:
words = []
labels = []
docs = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

words = [stemmer.stem(w) for w in words if w != '?']
words = sorted(list(set(words)))
labels = sorted(labels)


training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w) for w in doc]
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)

output = numpy.array(output)

with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)


tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)


   # model.load("model.tflearn")

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    #print(words)
    #print(s)
    s_words = nltk.tokenize.wordpunct_tokenize(s)
    s_words = [stemmer.stem(word) for word in s_words]
   # print(s_words)

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat():
    print("Enter your question and (quit) to stop!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        #print(bag_of_words(inp, words))
        results_index = numpy.argmax(results)
        #print(results)
        tag = labels[results_index]


        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg["responses"]
                print(fix_encoding(random.choice(responses)))
                break
       # print("عذراً! هل يمكنك تزويدي بالمزيد من المعلومات؟")




chat()
