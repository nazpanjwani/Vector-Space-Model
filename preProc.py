from pydoc import doc
from re import A
import re
from tempfile import tempdir
import os
from turtle import pos
# from nbformat import write
# from pyrsistent import b
import nltk
# nltk.download('omw-1.4')
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
import string
# from nltk.stem import PorterStemmer
# Pstem=PorterStemmer()
import numpy
# from scipy import spatial
from nltk.stem import WordNetLemmatizer
lemma=WordNetLemmatizer()
# nltk.download('wordnet')
def Inv_ind():
    global DIR 
    DIR = 'Abstracts/'
    i=0
    doc=[]
    index={}   #inverted index
    index_dup={}
    V=[]
    D_all=[]
    for i in range(448):
        docId=str(i+1)

        #reading content of files in abstract folder
        f1=open(DIR+docId+".txt",'r')
        file_con=f1.read()
        file_con=file_con.replace('\n',' ')

        file_con=re.sub(r'[0–9]+', '', file_con )
        f = open("Stopword-List.txt", 'r')
        for line in f:
            for st in line.split():
                file_con=file_con.lower().replace(' '+st+' ',' ')
        f.close()
    
        #Tokenizing words and removing punctuations and symbols
        doc=word_tokenize(file_con)
        doc=[''.join(c for c in s if c not in string.punctuation) for s in doc]
        doc=[s for s in doc if s]
        
        lemmatization=[]
        # for j in doc:
        #     lemmatization.append(lemma.lemmatize(j))

        # lemmatization of tokens
        # stemming=[]     
        for j in doc:
            lemmatization.append(lemma.lemmatize(j))
            # stemming.append(Pstem.stem(j))
        index_dup[i+1]=lemmatization

        #Removing duplicate tokens
        temp=[]
        for w in lemmatization: 
            D_all.append(w)
            if w not in temp:
                temp.append(w)
                V.append(w)

        index[i+1]=temp

    temp1=[]
    for w in V:
        if w not in temp1:
            temp1.append(w)
            # temp1.append(lemma.lemmatize(w))

    V=temp1
    num=[]
    for w in range(90):
        num.append(str(w))
        
    for w in V:
        if w in num:
            V.remove(w)

    return V, index, index_dup

def idf_tf(V, index, index_dup):

    global DIR 
    DIR = 'Abstracts/'
    i=0
    dict={}     #document content of all documents
    wgt={}
    df={}
    idf={}
    tf={}

    #calculating df
    for w in V:
        df.setdefault(w,0) 
        for key in index:
            if(w in index[key]):
                df[w]+=1

    #calculating idf
    for key in df:
        idf.setdefault(key,0)
        if(df[key]<=0):
            print(key)
            print(df[key])
        idf[key]=numpy.log2(448/df[key])

    #calculating tf
    for key in index_dup:
        dict.setdefault(key,{})
        for w in V:
            dict[key][w]=index_dup[key].count(w)
        if(448 == key):
            tf=dict

    #calculating tf*idf for each document
    for key in tf:
        wgt.setdefault(key,{})
        for w in V:
            wgt[key][w]=format(tf[key][w]*idf[w],'.3F')

    f = open("index.txt", "w")
    f.write(str(wgt))
    f.close()

    f = open("idf.txt", "w")
    f.write(str(idf))
    f.close()

    return idf, wgt

v,index, index_dup=Inv_ind()

f = open("V.txt", "w")
f.write(str(v))
f.close()
idf, d_wgt=idf_tf(v,index,index_dup)
