import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.corpus import stopwords
from string import punctuation
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix



df1 = pd.read_csv("../data/Duplicate_Bug_Report/Eclipse/EP_dup.csv", sep=";")
df2 = pd.read_csv("../data/Duplicate_Bug_Report/Eclipse/EP_nondup.csv", sep=";")
df3 = pd.read_csv("../data/Duplicate_Bug_Report/Mozilla/M_Duplicate BRs.csv", sep=";")
df4 = pd.read_csv("../data/Duplicate_Bug_Report/Mozilla/M_NonDuplicate BRs.csv", sep=";")
df5 = pd.read_csv("../data/Duplicate_Bug_Report/ThunderBird/dup_TB.csv", sep=";")
df6 = pd.read_csv("../data/Duplicate_Bug_Report/ThunderBird/Nondup_TB.csv", sep=";")
frames=[df1,df2,df3,df4,df5,df6]
df=pd.concat(frames)
print(df.shape)
data=df
data=df
data['Report1'] = data['Title1'].str.cat(data['Description1'],sep=" ")
data['Report2'] = data['Title2'].str.cat(data['Description2'],sep=" ")

data["Report1"].fillna( method ='ffill', inplace = True)
data["Report2"].fillna( method ='ffill', inplace = True)
data["Label"].fillna( method ='ffill', inplace = True)

data.isnull().sum()
a = data.pivot_table(index = ['Label'], aggfunc ='size')
a = a.reset_index()
a.columns= ["Values", "Counts"]

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
    return (text)

def process(bug_list, bugs):
    for bug in bugs:
        bug_list.append(words(bug))
processed_bug1 = []
processed_bug2 = []
process(processed_bug1, data.Report1)
process(processed_bug2, data.Report2)

a = 0
for i in range(a,a+10):
    print(processed_bug1[i])
    print(processed_bug2[i])



tfidf = TfidfVectorizer(analyzer='word', stop_words='english', lowercase=True, max_features=300,
                        norm='l1')
words = pd.concat([data.Report1, data.Report2], axis = 0)
tfidf.fit(words)
duplicate_1 = tfidf.transform(data.Report1)
duplicate_2 = tfidf.transform(data.Report2)
x = abs(duplicate_1 - duplicate_2)
y = data['Label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
lr = LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
TP = cm[1][1]
TN = cm[0][0]
FP = cm[1][0]
FN = cm[0][1]
print("True Positive  : ", TP)
print("True Negative  : ", TN)
print("False Positive : ", FP)
print("False Negative : ", FN)
print()

Accuracy = (TP + TN) / (TP + TN + FP + FN)
print("Accuracy: ",Accuracy)
accuracy_score(y_test, y_pred)

