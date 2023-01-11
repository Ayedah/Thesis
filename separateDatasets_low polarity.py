# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 20:37:41 2018

@author: aidaz
"""
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.utils import class_weight
from os import listdir
from os.path import isfile, join
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.contrib import learn
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import nltk
from keras.optimizers import Adam,Adadelta,Adagrad,SGD,RMSprop,Adamax,Nadam
import matplotlib.pyplot as plt
import spacy 
from sklearn.metrics import classification_report
from keras.models import Sequential,Model,model_from_json
#from keras.layers import Merge
from keras.layers import Dense,GlobalMaxPooling1D, Activation
from keras.layers import Flatten
from keras.layers import Embedding,Input,BatchNormalization
from keras.layers import concatenate
from keras.layers import Dropout
from keras.layers import MaxPooling1D
from keras.layers import Conv1D
from keras.layers import LSTM
from keras.utils import np_utils
from keras import optimizers
from gensim.models import Word2Vec
from numpy import random
import h5py
from keras.callbacks import Callback
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils import class_weight

df=pd.read_csv('comments-valence-3.csv.gz',header=0)
#df = pd.read_csv("Book1.csv",header=0)
df.head()
##df= df["text"].map(str) + df["senti.max.autotime"],df["senti.min.autotime"]
df=df.drop_duplicates(subset='text', keep='first')
df= df.dropna(subset=['text'])
#df= df.dropna(subset=['senti.min.autotime'])
#df= df.dropna(subset=['senti.max.autotime'])
#data1 =df.loc[df['senti.max.autotime'].isin([2,3]) & df['senti.min.autotime'].isin([-1])]
df['class'] = np.where(df['senti.max.autotime']>=3, 1, 0)
df.drop(['source','bug.id','comment.id','senti.min','senti.max','tensi.max','tensi.min','tensi.max.autotime','tensi.min.autotime'],axis=1,inplace=True)
count=len(df[(df['class']==0)])
print (count)
count=len(df[(df['class']==1)])
print (count)
df.to_csv('labeled_comments_valence3.csv')
