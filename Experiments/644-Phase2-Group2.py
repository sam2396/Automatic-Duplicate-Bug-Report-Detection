import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from string import punctuation


df1 = pd.read_csv("Data/Eclipse/EP_dup.csv", sep=";")
df2 = pd.read_csv("Data/Eclipse/EP_nondup.csv", sep=";")
df3 = pd.read_csv("Data/Mozilla/M_Duplicate BRs.csv", sep=";")  
df4 = pd.read_csv("Data/Mozilla/M_NonDuplicate BRs.csv", sep=";")
df5 = pd.read_csv("Data/ThunderBird/dup_TB.csv", sep=";")
df6 = pd.read_csv("Data/ThunderBird/Nondup_TB.csv", sep=";")

frames=[df1,df2,df3,df4,df5,df6]
data=pd.concat(frames)
data = data.reset_index(inplace = False)
data['TitleDescription1'] = data['Title1'].str.cat(data['Description1'],sep=" ")
data['TitleDescription2'] = data['Title2'].str.cat(data['Description2'],sep=" ")

stop_words = set(stopwords.words('english'))



def words(text, remove_stop_words=True, stem_words=False):
    # Remove punctuation from questions
    text = ''.join([c for c in text if c not in punctuation])
    
    # Lowering the words in questions
    text = text.lower()
    
    # Remove stop words from questions
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    
    # Return a list of words
    return(text)
    
    
    
    
def process(bug_list, bugs):
    for bug in bugs:
        bug_list.append(words(bug))
processed_bug1 = []
processed_bug2 = []
process(processed_bug1, data.TitleDescription1)
process(processed_bug2, data.TitleDescription2)

#Define tfidf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(analyzer = 'word',
                        stop_words = 'english',
                        lowercase = True,
                        max_features = 300,
                        norm = 'l1')
                        
#Concatenating TitleDescription Columns                        
words = pd.concat([data.TitleDescription1, data.TitleDescription2], axis = 0)

#Fitting and transforming bug reports with TFIDF Vectorizer
tfidf.fit(words)
duplicate_1 = tfidf.transform(data.TitleDescription1)
duplicate_2 = tfidf.transform(data.TitleDescription2)

#Assigning independent and dependent variables
x = abs(duplicate_1 - duplicate_2)
y = data['Label']

#Splitting into train and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# build model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D


model = Sequential()
model.add(Dense(64,input_shape=(300,)))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])
              
results = model.fit(x_train.todense(),y_train,batch_size=64,epochs=5,verbose=1,validation_data=(x_test.todense(),y_test))
print(np.mean(results.history["val_acc"]))


