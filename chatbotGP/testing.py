# -*- coding: utf-8 -*-
# encoding: utf-8
import nltk

nltk.download('punkt')
#from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.isri import ISRIStemmer

stemmer = ISRIStemmer()

import numpy
import tensorflow
import tflearn
import random
import json
import pickle
from ftfy import fix_encoding
import re
import sys

with open("intents.json" ,'r', encoding="utf-8") as file:
    data = json.load(file)

print(json.dumps(data, ensure_ascii = False))

for a in nltk.word_tokenize('في_بيتنا كل شي لما تحتاجه يضيع ...ادور على شاحن فجأة يختفي ..لدرجة اني اسوي نفسي ادور شيء '):
    print(stemmer.stem(a))
