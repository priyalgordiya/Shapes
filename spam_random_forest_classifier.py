# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 12:46:38 2024

@author: priya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
#-------------------------------------------Loading the data
df = pd.read_csv(r'C:\Users\priya\Downloads\spam.tsv', sep='\t')
#-------------------------------------------Understanding the data
df.head()
df.describe()
df['label'].value_counts()
#-------------------------------------------applying logics
ham = df[df['label']=='ham']
spam = df[df['label']=='spam']
ham = ham.sample(spam.shape[0])
data = ham.append(spam, ignore_index=True)
data.head()
#-------------------------------------------Creating Visualizations
plt.hist(data[data['label']=='ham']['length'], bins=100, alpha=0.7)
plt.hist(data[data['label']=='spam']['length'], bins=100, alpha=0.7)
plt.show()

plt.hist(data[data['label']=='ham']['punct'], bins=100, alpha=0.7)
plt.hist(data[data['label']=='spam']['punct'], bins=100, alpha=0.7)
plt.show()
#-------------------------------------------Training & Testing Data
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(data['message'], data['label'], test_size = 0.3, random_state = 0, shuffle=True)
X_Train.shape
#-------------------------------------------Vectorization & Random Forest Ensemble
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

classifier = Pipeline([("TfidfVectorizer", TfidfVectorizer()), ("Classifier", RandomForestClassifier(n_estimators=100))])
#-------------------------------------------Fitting the Training data
classifier.fit(X_Train, Y_Train)
#-------------------------------------------Y Prediction based on X testing
y_pred = classifier.predict(X_Test)
Y_Test, y_pred
#-------------------------------------------Reports, Matrix & Scores
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
a = accuracy_score(Y_Test, y_pred)*449
c = confusion_matrix(Y_Test, y_pred)
clr = classification_report(Y_Test, y_pred)
print(classification_report(Y_Test, y_pred))


#-------------------------------------------Testing the Data Model
Test1 = ['Hello, I am Nicolas']
Test2 = ['Hope you are doing good & learning new things']
Test3 = ['Congratulations!! you won a lottery ticket worth $1 Million. To claim call on 284288']

print(classifier.predict(Test1))
print(classifier.predict(Test2))
print(classifier.predict(Test3))


