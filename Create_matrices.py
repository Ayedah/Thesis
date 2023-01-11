# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:13:26 2018

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
import matplotlib.pyplot as plt
import spacy 
from keras.models import Sequential,Model,model_from_json
from keras.layers import Dense,GlobalMaxPooling1D, Activation
from keras.layers import Flatten
from keras.layers import Embedding,Input,BatchNormalization
from keras.layers import concatenate
from keras.layers import Dropout
from keras.layers import MaxPooling1D
from keras.layers import Conv1D
from keras.layers import LSTM
from keras.utils import np_utils
from gensim.models import Word2Vec
from numpy import random
import h5py
from keras.callbacks import Callback
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils import class_weight
import time
global find_tag
nlp = spacy.load("en_core_web_sm")


with tf.device('/gpu:0'):
    start = time.time()
    print("hello")

    df = pd.read_csv("NewTest.csv",header=0,sep=None)
    text= df['text'].as_matrix(columns=None)
    labelsN= df['labels'].as_matrix(columns=None)
    #when need to concatenate data from all csv files 
    #processed_docs_all = np.concatenate((processed_docs_train, processed_docs_test), axis=0)
    numWords = []
    #df_new = df[df['text'].notnull()]
    print (type(text))
    print(text.shape)
    for tweet in text:
            print(tweet)
            counter = len(tweet.split())
            numWords.append(counter)
            print(counter)
    labels=labelsN
    labels=labels.reshape((labels.shape[0], 1))
    print (labels.shape)
    #labels=np_utils.to_categorical(labels, 2)
    print(type(labels))
    print(labels.shape)
    num_labels = len(np.unique(labels))
    
    numFiles = len(numWords)
    print('The total number of words in the files is', sum(numWords))
    print('The average number of words in the files is', sum(numWords)/len(numWords))
    
    maxSeqLength = max([len(t.split(" ")) for t in text])
    print (maxSeqLength)
    #vocab_processor = learn.preprocessing.VocabularyProcessor(maxSeqLength)
    #x = np.array(list(vocab_processor.fit_transform(df.text)))
    #print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    #vocabulary_size=len(vocab_processor.vocabulary_)
    tokenizer = Tokenizer(filters='$%&()*/:;<=>@[\\]^`{|}~\t\n')
    tokenizer.fit_on_texts(text)
    vocab_size = len(tokenizer.word_index) + 1
    sequences = tokenizer.texts_to_sequences(text)
    data = pad_sequences(sequences, maxlen=maxSeqLength, padding='pre')
    print (data[0])
    print (len(data[0]))
    print (type(data))
    print (data.shape)
    print(data.shape[1])
    print (vocab_size)
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    
    #X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
    #print (X_train.shape, y_train.shape)
    #print (X_test.shape, y_test.shape)
    #print (X_train)
    
    #data_X=data[0:16,:]
    #print (data_X)
    filepath_glove = 'glove.twitter.27B.200d.txt'
    #filepath_glove = 'glove.6B.50d.txt'
    glove_vocab = []
    glove_embd=[]
    embedding_dict = {}
     
    file = open(filepath_glove,'r',encoding='UTF-8')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab_word = row[0]
        glove_vocab.append(vocab_word)
        embed_vector = [float(i) for i in row[1:]] # convert to list of float
        embedding_dict[vocab_word]=embed_vector
    file.close()
    
    
      
    print('Loaded GLOVE')
    
    #model = Word2Vec.load(C:\Users\aidaz\Sentiment analysis\tryout\'model.txt')
    
    from gensim.models.keyedvectors import KeyedVectors
    word_vect = KeyedVectors.load_word2vec_format("SO_vectors_200.bin", binary=True)
    
    
    embedding_matrix = np.zeros((vocab_size, 200))
    for word, index in tokenizer.word_index.items():     #it is looking at the indexes of data
        if index > vocab_size - 1:
            break
        else:
            try:
                embedding_vector = word_vect[word]
               # embedding_vector = embedding_dict[word]
            except KeyError:
                embedding_vector=None
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
            else:
                try:
                    embedding_vector = embedding_dict[word]
                except KeyError:
                     embedding_vector=None
                if embedding_vector is not None:
                    embedding_matrix[index] = embedding_vector
                else:
                    embedding_matrix[index] = np.random.uniform(-0.25,0.25,200)
                
    
    print (embedding_matrix.shape)
    print (embedding_matrix.shape[1])
    
    
    #binary bag as a dictionary where key is string and value is n-d hot vector         
    vocabulary = ['NAN','NIL','ADD','AFX','BES','CC', 'CD', 'DT','EX','FW','GW','HVS','HYPH','IN','JJ','JJR','JJS','LS','MD','NFP','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','_SP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB','XX',',','.',':','\'\'','""','#','``','$','-LRB-','-RRB-'] # Put your targer words in here
    hot_encoder = [0] * len(vocabulary)
    binary_bag = dict(zip(vocabulary, hot_encoder))
    binary_bag['NAN']=hot_encoder
    index=1
    for each in vocabulary[1:]:
        hot_encoder = [0] * len(vocabulary)
        hot_encoder[index]=1
        index=index+1
        (binary_bag[each])=hot_encoder
    
    print(binary_bag['NIL'])
    embedding_matrix_pos = np.zeros((len(vocabulary), len(vocabulary)))
    index=0
    for each in vocabulary:
        print(each)
        embedding_vector_pos = binary_bag[each]
        embedding_matrix_pos[index] = embedding_vector_pos
        index=index+1
        
    print(embedding_matrix_pos[1])
    print(type(embedding_matrix_pos))
    #print(type(embedding_matrix))  
    
    
    number=0
    stringToInt = {}
    for each in vocabulary:
        stringToInt[each]=number
        print(number)
        number=number+1
    
    #embedding pos has each sentence in POS tags(string e.g. NN)
    embedding_pos=[]
    index=0
    for tweet in text:
        t=nlp(tweet)
        pos_tags = [(i, i.tag_) for i in t] 
        temp=dict(pos_tags).values()
        number=len(temp)
        embedding_pos.append([])
        for i in range(number,maxSeqLength):
            #embedding_pos[index].append('NAN')
            embedding_pos[index].append(0)
        for each in temp:
                embedding_pos[index].append(stringToInt[each])
        index=index+1

    embedding_pos=np.asarray(embedding_pos)         
    end = time.time()
    print(end - start)
    print(embedding_matrix.shape)
    print(embedding_matrix_pos.shape)
    print(type(embedding_pos)) 
    print(embedding_pos.shape)
    print(type(data))
    print(data.shape) 
    print(data[4])
    print(embedding_pos[4])
    with h5py.File('SavedDataShuffledNewTest2.h5', 'w') as hf:
        hf.create_dataset("sequence_data",  data=data)
        hf.create_dataset("data_Matrix",  data=embedding_matrix)
        hf.create_dataset("sequence_POS",  data=embedding_pos)
        hf.create_dataset("POS_Matrix",  data=embedding_matrix_pos)
        hf.create_dataset("Labels",  data=labels)      
        hf.create_dataset("maxSeqLength", data=maxSeqLength)
        hf.create_dataset("vocab_size", data=vocab_size)