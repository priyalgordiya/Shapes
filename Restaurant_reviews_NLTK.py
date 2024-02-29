# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 11:00:12 2024

@author: priya
"""


import pandas as pd
import numpy as np
#-------------------------------------------Loading the data
data = pd.read_csv(r'C:\Users\priya\Downloads\Restaurant_Reviews.tsv', sep='\t', quoting=3)
#-------------------------------------------Checking the data
data['Liked'].value_counts() #to check the balance of the data

#-------------------------------------------Cleaning & Pre-Processing Data
import nltk
import re

nltk.download('stopwords')
from nltk.corpus import stopwords
#-------------------------------------------Removing quotes, dots & lower case
review = re.sub('[^a-zA-Z]', ' ', data['Review'][0])
review = review.lower()
#-------------------------------------------Tokenization
review = review.split()
#-------------------------------------------reviewing stopwords
stopwords.words('english')

preview = []
for word in review:
    if not word in stopwords.words('english'):
        preview.append(word)
        
review = [word for word in review if not word in stopwords.words('english')]
#-------------------------------------------Stemming
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
review = [ps.stem(word) for word in review]   
review = " ". join(review) 
#-------------------------------------------Apply Tokenization & Stemming in all reviews
corpus = []
ps = PorterStemmer()
for i in range(len(data)):
    review = re.sub('[^a-zA-Z]', ' ', data['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = " ". join(review) 
    corpus.append(review)
#-------------------------------------------Feature Engineering/Bag of word Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()
y = data.iloc[:, 1].values
#-------------------------------------------Naive Bayes Algorithm
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(x, y, test_size = 0.20, random_state = 0, shuffle=True)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_Train, Y_Train)
y_pred = classifier.predict(X_Test)
#--------------------------------------------Checking accuracy
from sklearn.metrics import accuracy_score
ac = accuracy_score(Y_Test, y_pred)*200
acs_per = accuracy_score(Y_Test, y_pred)

