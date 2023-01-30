from re import A
import re
import os
from tkinter import messagebox
import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
import string
# from nltk.stem import PorterStemmer
# Pstem=PorterStemmer()
import tkinter as tk
from tkinter import simpledialog
from tkinter import messagebox
import numpy
from scipy import spatial
from nltk.stem import WordNetLemmatizer
lemma=WordNetLemmatizer()
import ast
root=tk.Tk()
root.withdraw()

#Query processing
def query_process(query, V, idf):
    wgt={}
    tf={}
    token=[]
    stemming=[]
    token=word_tokenize(query.lower())
    token=[''.join(c for c in s if c not in string.punctuation) for s in token]
    token=[s for s in token if s]
    for w in token:
        # stemming.append(Pstem.stem(w))
        stemming.append(lemma.lemmatize(w))

    for w in V:
        tf.setdefault(w,0)
        tf[w]=stemming.count(w)

    for w in V:
        wgt.setdefault(w,0)
        wgt[w]=format(tf[w]*idf[w],'.3F')
    return(wgt)

#calculating cosine sim of (q,di)
def cos_sim(d_wgt, q_wgt):
    cos={}
    for key in d_wgt:
        temp=[]
        temp1=[]
        for w in q_wgt:
            temp.append(float(d_wgt[key][w]))
            temp1.append(float(q_wgt[w]))
        cos.setdefault(key,0)
        cos[key]=format(1-spatial.distance.cosine(temp,temp1),'.3F')
    return cos

#Retrieving documents
def get_docs(cos):
    docs={}
    ans=[]
    alpha=0.001
    for key in cos:        
        if(float(cos[key])>alpha):
            docs.setdefault(key,0)
            docs[key]=cos[key]
    for key in docs:
        ans.append(key)
    messagebox.showinfo("Documents matching this Query: ",ans)

f= open("index.txt", 'r')
tmp=f.read()
d_wgt = ast.literal_eval(tmp)
f.close()

f= open("idf.txt", 'r')
tmp=f.read()
idf = ast.literal_eval(tmp)
f.close()

f= open("idf.txt", 'r')
tmp=f.read()
v = ast.literal_eval(tmp)
f.close()

query=simpledialog.askstring(title="Vector Space Model", prompt="Enter Query: ")
q_wgt=query_process(query, v, idf)
cos=cos_sim(d_wgt,q_wgt)
get_docs(cos)