from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from pprint import pprint


newsgroups_train = fetch_20newsgroups(subset='train')
label = newsgroups_train.target
print('number of docs is:', len(label))
np.save('../data/document/label.npy',label)



vectorizer = TfidfVectorizer(analyzer='word')
vectors = vectorizer.fit_transform(newsgroups_train.data)
vocabulary = vectorizer.get_feature_names_out()
tf = vectors.toarray()
tf = np.sum(tf,0)
idx = np.argsort(tf)[::-1]
idx = idx[0:3000]
print('number of words are:', len(idx))



vectorizer = CountVectorizer()
X = vectorizer.fit_transform(newsgroups_train.data)
tf = X.toarray()
print(np.shape(tf))
X = tf[:,idx]
np.save('../data/document/X.npy',X)




#     X = vectorizer.fit_transform(newsgroups.data)
#     tf = X.toarray()
#     print(np.shape(tf))
    
#     #print(np.sum(tf,1))

# #cats = newsgroups.target_names
# #newsgroups = fetch_20newsgroups(subset='train', categories=newsgroups.target_names)
# #X = vectorizer.fit_transform(newsgroups.data)

# #print( vectorizer.get_feature_names_out()  )
# #print(X)
# #tf = X.toarray()
# #print(np.shape(tf))
# #print(cats)
# # for cat in cats:
# #     print(cat)
# #     newsgroups_cat = fetch_20newsgroups(subset='train', categories=[cat])
    
#     #print(np.sum(tf,1))
    




# #print(idx)
# #print(np.shape(tf))
# #print(tf[idx])

# #vectorizer = CountVectorizer()
# #X = vectorizer.fit_transform(newsgroups_train.data)
# #
# #features = vectorizer.get_feature_names_out()
# #print("Feature names:\n", features)
# #print("Term Frequencies:\n", np.min(tf[tf > 1]))
# # vectors = vectorizer.fit_transform(newsgroups_train.data)
# # print(vectors)