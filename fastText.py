import numpy as np 
import pandas as pd 
import matplotlib.pylab as plt
import json, argparse, os
from pathlib import Path
import fasttext

label2category = {0: "POLITICS", 1: "ENTERTAINMENT", 2: "HEALTHY LIVING", 3: "QUEER VOICES", 4: "BUSINESS",
                  5: "SPORTS", 6: "COMEDY", 7: "PARENTS", 8: "BLACK VOICES", 9: "THE WORLDPOST",
                  10: "WOMEN", 11: "CRIME", 12: "MEDIA", 13: "WEIRD NEWS", 14: "GREEN", 15: "IMPACT",
                  16: "WORLDPOST", 17: "RELIGION", 18: "STYLE", 19: "WORLD NEWS", 20: "TRAVEL", 21: "TASTE",
                  22: "ARTS", 23: "FIFTY", 24: "GOOD NEWS", 25: "SCIENCE", 26: "ARTS & CULTURE", 27: "TECH",
                  28: "COLLEGE", 29: "LATINO VOICES", 30: "EDUCATION"}

category2label = {"POLITICS" :0, "ENTERTAINMENT": 1, "HEALTHY LIVING": 2, "QUEER VOICES": 3, "BUSINESS": 4,
                  "SPORTS": 5, "COMEDY": 6, "PARENTS": 7, "BLACK VOICES": 8, "THE WORLDPOST": 9,
                  "WOMEN": 10, "CRIME": 11, "MEDIA": 12, "WEIRD NEWS": 13, "GREEN": 14, "IMPACT": 15,
                  "WORLDPOST": 16, "RELIGION": 17, "STYLE": 18, "WORLD NEWS": 19, "TRAVEL": 20, "TASTE": 21,
                  "ARTS": 22, "FIFTY": 23, "GOOD NEWS": 24, "SCIENCE": 25, "ARTS & CULTURE": 26, "TECH": 27,
                  "COLLEGE": 28, "LATINO VOICES": 29, "EDUCATION": 30}

data = Path("News_Category_Dataset.json")
training_data = Path("fasttext_dataset_train.txt")
test_data = Path("fasttext_dataset_test.txt")

#Percentage of data in train and test set
percent_test_data = 0.25

#Preprocessing and Formatting
import re
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
ps = PorterStemmer()

def strip_formatting(string):
    string = string.lower()
    string = re.sub('[^a-zA-Z]', ' ', string)
    string = string.split()
    string = [ps.stem(word) for word in string if not word in set(stopwords.words('english'))]
    string = ' '.join(string)
    
    return(string)
    
with data.open() as input, \
     training_data.open("w") as train_output, \
     test_data.open("w") as test_output:

    for line in input:
        news_data = json.loads(line)

        label = news_data['category']
        news_data['text'] = news_data['headline'] + " " + news_data['short_description']
        text = news_data['text'].replace("\n", " ")
        text = strip_formatting(text)

        fasttext_line = "__label__{} {}".format(label, text)

        if random.random() <= percent_test_data:
            test_output.write(fasttext_line + "\n")
        else:
            train_output.write(fasttext_line + "\n")
            
            
#Model Building
            
classifier = fasttext.load_model('model.bin', encoding = 'utf-8')

model = fasttext.skipgram('fasttext_dataset_train.txt', 'model')

classifier = fasttext.supervised(input_file="fasttext_dataset_train.txt",
                                     output='model',
                                     dim = 300,
                                     lr=0.01,
                                     epoch = 30)
                                     

result = classifier.test('fasttext_dataset_test.txt')
print ('P@1:', result.precision)
print ('R@1:', result.recall)
print ('Number of examples:', result.nexamples)

texts = ['former disney exec launch social network girl maverick aim connect empow young women girl']
print (classifier.predict(texts, k=10))

#Improve Models
classifier_2 = fasttext.supervised(input_file="fasttext_dataset_train.txt",
                                     output='model',
                                     dim = 500,
                                     lr=0.01,
                                     epoch = 20)

result_2 = classifier_2.test('fasttext_dataset_test.txt')
print ('P@1:', result_2.precision)
print ('R@1:', result_2.recall)
print ('Number of examples:', result_2.nexamples)

resultt = classifier.test('fasttext_dataset_train.txt')
print ('P@1:', resultt.precision)
print ('R@1:', resultt.recall)
print ('Number of examples:', resultt.nexamples)