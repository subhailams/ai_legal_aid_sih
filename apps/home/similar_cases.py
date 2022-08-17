import pickle
import pandas as pd
from rank_bm25 import BM25Okapi

from django.conf import settings

import re
import nltk
lst_stopwords = nltk.corpus.stopwords.words("english")

ps = nltk.stem.porter.PorterStemmer()

nltk.download('wordnet')
lem = nltk.stem.wordnet.WordNetLemmatizer()

open_file = open(settings.MODEL_ROOT + "/tokenized_corpus", "rb")
tokenized_corpus = pickle.load(open_file)
n=pd.read_csv(settings.MODEL_ROOT + "/corpus_names.csv")
bm25 = BM25Okapi(tokenized_corpus)

def utils_preprocess_text(text, flg_stemm=True, flg_lemm =True, lst_stopwords=None ):
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    
    #tokenization(convert from string to List)
    lst_text = text.split()
    
    #remove stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in
                   lst_stopwords]
        
     #stemming
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
        
    #Lemmentization
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
        
    # back to string from list
    text = " ".join(lst_text)
    return text

def similarcase(a):
     b = utils_preprocess_text(a, flg_stemm = False, flg_lemm=True)
     open_file.close()
          
     name=n["Name"]
     print(len(tokenized_corpus))
     print(len(name))
     print(bm25.get_top_n(b.split(" "),name, n=10))
     return bm25.get_top_n(b.split(" "),name, n=10)

# user_text=""
# with open("C2.txt",'r') as f:
#     for line in f.readlines():
#         user_text+=line.strip()
# similarcase(user_text)