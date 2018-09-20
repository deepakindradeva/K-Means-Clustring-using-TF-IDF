import os
import glob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

file_list = glob.glob(os.path.join(os.getcwd(), "C:\\Users\\DeepakIndraDeva\\Desktop\\BCA\\Minor_Project\\2nd_Assignment\\Documents", "*.txt"))

corpus = []

for file_path in file_list:
    with open(file_path) as f_input:
        corpus.append(f_input.read())

tfidf = TfidfVectorizer()

tfs = tfidf.fit_transform(corpus)

df=pd.DataFrame(tfs.toarray())

df.to_csv('C:\\Users\\DeepakIndraDeva\\Desktop\\out.csv', sep=',')