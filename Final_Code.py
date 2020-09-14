import math
import glob
import os
from textblob import TextBlob as tb
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def tf(word, blob):
    return blob.split().count(word) / len(blob.split())

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.split())

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)


folder = "C:\\Users\\DeepakIndraDeva\\Desktop\\BCA\\Major_Project\\Project\\DATA"
os.chdir(folder)
files = glob.glob("*.txt") # Makes a list of all files in folder
bloblist = []
for file1 in files:
  with open (file1, 'r') as f:
   data = f.read() # Reads document content into a string
   document = tb(data)
   bloblist.append(document) 
   
   
#cleaning the data
i = 0
for blob in bloblist:
    line = ""
    ps = PorterStemmer()
    for word in  blob.words :
        word = word.lower()
        word = word.strip()
        if len(word)>3:
            if ps.stem(word) not in stopwords.words('english'):
                line += word +' '        
    line = re.sub(',',' ',line)
    line = re.sub(r'^https?:\/\/.*[\r\n]*', '', line, flags=re.MULTILINE)
    line = re.sub('[^A-Za-z .-]+', ' ', line)
    line = re.sub('[^a-zA-Z#]', ' ', line)
    line = line.replace('-', '')
    line = line.replace('…', '')
    line = line.replace('  ', '')
    bloblist[i] = line
    i = i+1


#total is a blank string where we take words of all rows 
total = " "
for blob in bloblist:
  for word in blob:
    total = total + word
 
    
#sorting all words as per their tf-idf scores      
scores_all = {word: tfidf(word, blob, bloblist) for word in total.split()}
sorted_words = sorted(scores_all.items(), key=lambda x: x[1], reverse=True)


#taking top 350 of them
store = []
for word in sorted_words[:350]:                
      store.append(word)      


#taking a blank dataframe    
data = np.array([np.arange(119)*1]).T
X = pd.DataFrame(data)         


#creating an output matrix 
l = 1
for blob in bloblist:
        scores = {word: tfidf(word, blob, bloblist) for word in blob.split()}
        score_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for word,score in score_words[:10]:
            for j in range(0,350):
                if (word == store[j][0]):
                   X.at[l,word] = round(score, 6)
        l = l+1            
X.fillna(0 , inplace = True)        


#storing into file
y = X.iloc[:,1:]
y.to_csv("C:\\Users\\DeepakIndraDeva\\Desktop\\BCA\\Major_Project\\Project\\result31.csv",index = True)


   
df = pd.read_csv('C:\\Users\\DeepakIndraDeva\\Desktop\\BCA\\Major_Project\\Project\\result31.csv')


# For K-Means Clustering
from sklearn.cluster import KMeans
km = KMeans(n_clusters=6)
km.fit(df)
centroids = km.cluster_centers_
labels = km.labels_
#print(centroids)
#print(labels)


# for plot styling
import matplotlib.pyplot as plt
centers = km.cluster_centers_
plt.figure(figsize=(10,10))
for i in range(1,68):
    plt.scatter(df.values[:,0],df.values[:,i],s=50 , c = labels)
#plt.scatter(centers[:, 0], centers[:, 1], marker='X', c='black');
# Draw white circles at cluster centers
plt.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=50, edgecolor='k')
for i, c in enumerate(centers):
    plt.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,s=50, edgecolor='k')
plt.show()

# For Silhouette-score

from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
range_n_clusters = [2, 3, 4, 5, 6]
for n_clusters in range_n_clusters:
    fig,ax1 = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(df) + (n_clusters + 1) * 10])
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(df)
    silhouette_avg = silhouette_score(df, cluster_labels)
    print("For n_clusters =", n_clusters,"The average silhouette_score is :", silhouette_avg)
    sample_silhouette_values = silhouette_samples(df, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),0, ith_cluster_silhouette_values,facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
plt.show()
