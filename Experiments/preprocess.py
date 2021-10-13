import nltk
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')


df1 = pd.read_csv("Eclipse/EP_dup.csv", sep=";")
df2 = pd.read_csv("Eclipse/EP_nondup.csv", sep=";")
df3 = pd.read_csv("Mozilla/M_Duplicate BRs.csv", sep=";")
df4 = pd.read_csv("Mozilla/M_NonDuplicate BRs.csv", sep=";")
df5 = pd.read_csv("ThunderBird/dup_TB.csv", sep=";")
df6 = pd.read_csv("ThunderBird/Nondup_TB.csv", sep=";")
frames=[df1,df2,df3,df4,df5,df6]
df=pd.concat(frames)
print(df.shape)

df.drop(['Issue_id','Duplicated_issue','Label'],axis=1, inplace=True)
original_columns=['Title1','Description1','Title2','Description2']
for col in original_columns:
    new_col=col+"_tokenize"
    df[new_col] = df[col].apply(lambda x:word_tokenize(x))

columns=['Title1_tokenize','Description1_tokenize','Title2_tokenize','Description2_tokenize']

stemming=PorterStemmer()
for col in columns:
     df[col] = df[col].apply(lambda x: [stemming.stem(y) for y in x])

stops = set(stopwords.words("english"))
for col in columns:
     df[col] = df[col].apply(lambda x: [item for item in x if item not in stops])


for col in columns:
    df[col] = df[col].astype(str).str.lower()

print(df.head())