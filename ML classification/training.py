# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string

from google.colab import drive
drive.mount('/content/drive')

d_fake=pd.read_csv("Fake.csv")
d_true=pd.read_csv("True.csv")

d_fake.head()

d_fake["class"]=0
d_true["class"]=1

d_fake.shape, d_true.shape

d_fake_manual_testing = d_fake.tail(10)
for i in range(23480, 21470, -1):
    d_fake.drop([i], axis=0, inplace=True)

d_true_manual_testing = d_true.tail(10)
for i in range(21416, 21406, -1):
    d_true.drop([i], axis=0, inplace=True)

d_fake.shape, d_true.shape

d_fake_manual_testing['class'] = 0
d_true_manual_testing['class'] = 1

d_fake_manual_testing.head(10)

d_true_manual_testing.head(10)

d_merge = pd.concat([d_fake, d_true], axis = 0)
d_merge.head(10)

d_merge.columns

d = d_merge.drop(['title', 'subject', 'date'], axis=1)

d.isnull().sum()

d=d.sample(frac = 1)

d.head()

d.reset_index(inplace = True)
d.drop(['index'], axis=1, inplace = True)

d.columns

d.head()

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*/]', '', text)
    text = re.sub("\\W", " ", text)
    text + re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' %re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('w*\d\w*', '', text)
    return text

d['text'] = d['text'].apply(wordopt)

x = d['text']
y = d['class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(xv_train, y_train)

pred_lr = LR.predict(xv_test)

LR.score(xv_test, y_test)

print(classification_report(y_test, pred_lr))

from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)

pred_dt = DT.predict(xv_test)

DT.score(xv_test, y_test)

print(classification_report(y_test, pred_dt))

from sklearn.ensemble import GradientBoostingClassifier

GB = GradientBoostingClassifier(random_state = 0)
GB.fit(xv_train, y_train)

predict_gb = GB.predict(xv_test)

GB.score(xv_test, y_test)

print(classification_report(y_test, predict_gb))

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(random_state = 0)
RF.fit(xv_train, y_train)

pred_rf = RF.predict(xv_test)
RF.score(xv_test, y_test)

print(classification_report(y_test, pred_rf))

def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n==1:
        return "Not a Fake News"

def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transformation(new_x_test)
    pred_LR = LR.predoct(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GB.predict(new_xv_test)
    pred_RFC = RF.predict(new_xv_test)

    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(output_lable(pred_LR[0]),
                                                                                                             output_lable(pred_GB[0]),
                                                                                                             output_lable(pred_RF[0])))

news = str(input())
manual_testing(news)
