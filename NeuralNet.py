# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:22:47 2018

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
from sklearn.metrics import roc_curve,auc
import h5py
from keras.callbacks import Callback
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils import class_weight
import time
global find_tag

with h5py.File('SavedDataShuffled.h5', 'r') as hf:
    data = hf['sequence_data'][:]
    data= np.array(data)
    embedding_matrix= hf['data_Matrix'][:]
    embedding_matrix= np.array(embedding_matrix)
    embedding_pos= hf['sequence_POS'][:]
    embedding_pos = np.array(embedding_pos)
    embedding_matrix_pos= hf['POS_Matrix'][:]
    embedding_matrix_pos = np.array(embedding_matrix_pos)
    labels= hf['Labels'][:]
    labels = np.array(labels)
    print(labels.shape)
    print(embedding_pos.shape)
    print(embedding_pos[4])
    #maxSeqLength=hf['maxSeqLength']
    #vocab_size=hf['vocab_size']
with tf.device('/gpu:0'):
    
    maxSeqLength=267
    vocab_size=156451
    
    
        
        #also need to load maxSeqLen and vocab_size
        
    class_weights = {1: 1.0, 0: labels[0:607373,:][labels[0:607373,:] == 1].size / labels[0:607373,:][labels[0:607373,:] == 0].size}
    #class_weight_list = class_weight.compute_class_weight('balanced', np.unique(labels[0:8,:]),labels[0:8,:])
    #class_weights = dict(zip(np.unique(labels[0:8,:]), class_weight_list))
    
    #sm = SMOTE(random_state=12, ratio = 'auto', kind = 'regular')  #using SMOTE for balancing imbalanced data
    
    #kx_train_res, ky_train_res = sm.fit_sample(data[0:8,:], np.ravel(labels[0:8,:]))
    
    def generator(data,embedding_pos, labels, batch_size):
     # Create empty arrays to contain batch of features and labels#
     print (labels)
     batch_data = np.zeros((batch_size, maxSeqLength))
     batch_embedding_pos = np.zeros((batch_size, maxSeqLength))
     batch_labels = np.zeros((batch_size,1))
     num=len(data)
     while True:
       for i in range(batch_size):
         # choose random index in features
         index= random.choice(num,1)
         batch_data[i] = data[index]
         batch_labels[i]=labels[index]
         batch_embedding_pos[i] = embedding_pos[index]
         print(batch_labels)
       yield ([batch_data,batch_embedding_pos], batch_labels)
    
    
    class Metrics(Callback):
        def on_train_begin(self, logs={}):
            self.val_f1s = []
            self.val_recalls = []
            self.val_precisions = []
            self.val_f1sM = []
            self.val_recallsM = []
            self.val_precisionsM = []
            #self.val_accuracy_pos = []
            #self.val_accuracy_neg = []
         
        def on_epoch_end(self, epoch, logs={}):
            #print (self.validation_data[1])
            val_predict = (np.asarray(self.model.predict([self.validation_data[0],self.validation_data[1]]))).round()
            print (val_predict)
            val_targ = self.validation_data[2]
            print (val_targ)
            #_val_f1 = f1_score(val_targ, val_predict)
            #_val_recall = recall_score(val_targ, val_predict)
            #_val_precision = precision_score(val_targ, val_predict)
            _val_precisionM,_val_recallM,_val_f1M,support=precision_recall_fscore_support(val_targ, val_predict, average='macro')
            self.val_f1sM.append(_val_f1M)
            self.val_recallsM.append(_val_recallM)
            self.val_precisionsM.append(_val_precisionM)
            _val_precision,_val_recall,_val_f1,support=precision_recall_fscore_support(val_targ, val_predict, average=None , labels=[1,0])
            self.val_f1s.append(_val_f1)
            self.val_recalls.append(_val_recall)
            self.val_precisions.append(_val_precision)
            '''tn, fp, fn, tp = confusion_matrix(val_targ,val_predict).ravel()
            accuracy1=((tp+tn)/(tp+fn+tn+fp))
            accuracy2=((tn+tp)/(tn+fp+fn+tp))
            cmat = confusion_matrix(val_targ,val_predict,labels=[1,0])
            ch=cmat.diagonal()/cmat.sum(axis=1)
            self.val_accuracy_pos.append(ch[0])
            self.val_accuracy_neg.append(ch[1])
            print (' — val_accu_pos:',accuracy1, '— val_accu_neg:' ,accuracy2)
            print (' — val_accu_pos:', ch[0],'— val_accu_neg:' ,ch[1])'''
            print (' — val_f1:',_val_f1, '— val_precision:' ,_val_precision, '— val_recall :' ,_val_recall )
            return
    metrics = Metrics()
    class NeuralNetwork:
        def __init__(self):
            self._model = None
        
        def load_weights(self, *args, **kwargs):
            return self._model.load_weights(*args, **kwargs)
        
        def fit_generator(self, *args, **kwargs):
            return self._model.fit_generator(*args, **kwargs)
        
        def predict(self, *args, **kwargs):
            return self._model.predict(*args, **kwargs)
        @staticmethod
        def probas_to_classes(proba):
            if proba.shape[-1] > 1:
                return proba.argmax(axis=-1)
            else:
                return (proba > 0.5).astype('int32')
        def Save_model(self, *args, **kwargs):
            return self._model.predict(*args, **kwargs)
           
    class CNN(NeuralNetwork):
        def __init__(self, model):
            self._model = model
    #model_glove = Sequential()
        @classmethod
        def createCNN(self):
            inp = Input(shape=(data.shape[1],))
            e= Embedding(vocab_size, 200, input_length=maxSeqLength, weights=[embedding_matrix], trainable=False)(inp)
            conv1_1 = Conv1D(filters=100, kernel_size=2)(e)
            print(conv1_1.shape)
            btch1_1 = BatchNormalization()(conv1_1)
            actv1_1 = Activation('relu')(btch1_1)
            print(actv1_1.shape)
            glmp1_1 = GlobalMaxPooling1D()(actv1_1)
            
            '''conv1_3 = Conv1D(filters=32, kernel_size=4)(e)
            print(conv1_3.shape)
            btch1_3 = BatchNormalization()(conv1_3)
            actv1_3 = Activation('relu')(btch1_3)
            print(actv1_3.shape)
            glmp1_3 = GlobalMaxPooling1D()(actv1_3)
            
            #model_title = Sequential()'''
            inp2 = Input(shape=(embedding_pos.shape[1],))
            e1= Embedding(57, 57, input_length=maxSeqLength, weights=[embedding_matrix_pos], trainable=False)(inp2)
            
            conv1_2 = Conv1D(filters=100, kernel_size=2)(e1)
            btch1_2 = BatchNormalization()(conv1_2)
            actv1_2 = Activation('relu')(btch1_2)
            glmp1_2 = GlobalMaxPooling1D()(actv1_2)
            print (e.shape)
            print(e1.shape)
            #mer=concatenate([e,e1],axis=-1)
            #print (mer.shape)
            #lstm_out=LSTM(32)(mer)

            cnct = concatenate([glmp1_1,glmp1_2], axis=1)
            drp1_1  = Dropout(0.25)(cnct)
   
            dns1  = Dense(30, activation='relu')(drp1_1)
            drp2  = Dropout(0.10)(dns1)
            out=(Dense(1, activation='sigmoid')(drp2))
            
            
            model = Model(inputs=[inp, inp2], outputs=out)
            #opt=Adamax(lr=0.004)
            #opt=Adadelta(lr=0.001)
            #keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
            model.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['acc'])
            print(model.summary())
            return model
        
        def Save_model(self):
            model_json = model.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(model_json)
    # serialize weights to HDF5
            model.save_weights("modeltest.h5")
            print("Saved model to disk")
        def load_model(self):
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
            loaded_model.load_weights("modeltest.h5")
            print("Loaded model from disk")
    
    
       
    #train
    #first create object of CNN model which will return it
    #model=CNN()
    
    model=CNN.createCNN()
    #history=model.fit_generator(generator(data[0:8,:],embedding_pos[0:8,:],labels[0:8,:], batch_size=4),steps_per_epoch=8/4, samples_per_epoch=8, nb_epoch=10,validation_data = generator(data[8:10,:],embedding_pos[8:10,:],labels[8:10,:], batch_size=1),validation_steps = 2/1,use_multiprocessing=False,shuffle=False,class_weight=class_weight,workers=multiprocessing.cpu_count(),max_queue_size=10)
    start = time.time()
    print("hello")
    history=model.fit([data[0:607373,:],embedding_pos[0:607373,:]], labels[0:607373,:], batch_size=32,  epochs=20, shuffle=True,validation_data=([data[607373:729550,:],embedding_pos[607373:729550,:]],labels[607373:729550,:]),callbacks=[metrics],verbose=2,class_weight=class_weights)
    # summarize history for accuracy
    #history.callbacks
    end = time.time()
    print(end - start)
    x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    plt.plot(x,history.history['acc'])
    plt.plot(x,history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.show()
    #with h5py.File('Savedlogs.h5', 'a') as hf:
        #hf.create_dataset("Adamax_accuracy",  data=history.history['acc'])
        #hf.create_dataset("Adamax_val_accuracy",  data=history.history['val_acc'])
       
       
    # summarize history for loss
    
    plt.plot(x,history.history['loss'])
    plt.plot(x,history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.show()
    
    plt.plot(x,metrics.val_precisionsM)
    plt.title('Model Precision:Macro-Avg')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.show()
    
    plt.plot(x,metrics.val_f1sM)
    plt.title('Model F1:Macro-Avg')
    plt.ylabel('F1')
    plt.xlabel('epoch')
    plt.show()
    
    plt.plot(x,metrics.val_recallsM)
    plt.title('Model Recall:Macro-Avg')
    plt.ylabel('recall')
    plt.xlabel('epoch')
    
    plt.show()
    for Y in [metrics.val_precisionsM,metrics.val_recallsM,metrics.val_f1sM]:
        plt.plot(x,Y)
    plt.title('Model Precision-recall-F1:Macro-Avg')
    plt.ylabel('P-R-F1')
    plt.xlabel('epoch')
    plt.legend(['precision', 'recall','F1'], loc='upper left')
    plt.show()
    
    
    plt.plot(x,metrics.val_precisions)
    plt.title('Model Precision')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.legend(['positive', 'negative'], loc='upper left')
    plt.show()
    
    plt.plot(x,metrics.val_f1s)
    plt.title('Model F1')
    plt.ylabel('F1')
    plt.xlabel('epoch')
    plt.legend(['positive', 'negative'], loc='upper left')
    plt.show()
    plt.title('Model Recall')
    plt.ylabel('recall')
    plt.xlabel('epoch')
    plt.plot(x,metrics.val_recalls)
    plt.legend(['positive', 'negative'], loc='upper left')
    plt.show()
    
    
    loss, accuracy = model.evaluate([data[729550:878976],embedding_pos[729550:878976]],labels[729550:878976], verbose=1)
    print('Accuracy: %f' % (accuracy*100))
    print('Loss: %f' % (loss* 100))
    test_predict =(np.asarray( model.predict([data[729550:878976], embedding_pos[729550:878976]]))).round()
    test_target =labels[729550:878976]
    '''cmat = confusion_matrix(test_target,test_predict,labels=[1,0])
    print (cmat)
    print(cmat.diagonal()/cmat.sum(axis=1))'''
    tn, fp, fn, tp = confusion_matrix(test_target,test_predict).ravel()
    print (tp)
    print (tn)
    print (fp)
    print (fn)
    fpr_rf, tpr_rf, thresholds_rf = roc_curve(test_target, test_predict)
    auc_rf = auc(fpr_rf, tpr_rf)
    plt.plot(fpr_rf, tpr_rf, label='CNN (area = {:.3f})'.format(auc_rf))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    _test_precision,_test_recall,_test_f1,support=precision_recall_fscore_support(test_target,test_predict, average='macro')
    print (' — test_f1:',_test_f1, '— test_precision:' ,_test_precision, '— test_recall :' ,_test_recall )
    _test_precision1,_test_recall1,_test_f11,support=precision_recall_fscore_support(test_target,test_predict, average=None,labels=[1,0])
    print (' — test_f1:',_test_f11, '— test_precision:' ,_test_precision1, '— test_recall :' ,_test_recall1 )
    _test_precision,_test_recall,_test_f1,support=precision_recall_fscore_support(test_target,test_predict, average='micro')
    print (' — test_f1:',_test_f1, '— test_precision:' ,_test_precision, '— test_recall :' ,_test_recall )
    model_json = model.to_json()
    with open("f.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("f.h5")
    print("Saved model to disk")
    #model.save()
    '''model.Save_model()
    #model=model.load_model()
    #prediction = model.predict(input the pos test sequences as well as pos sequences)
    #print(prediction)
    left_branch = Sequential()
    left_branch.add(e)
    right_branch = Sequential()
    right_branch.add(e1)
    merged = Merge([left_branch, right_branch], mode='concat')
    model_embed = Sequential()
    model_embed.add(merged)
    model_embed = Sequential()
    model_embed.add(merged_input)
    model_embed.add(Flatten())
    model_embed.add(Dense(1, activation='sigmoid'))
    model_embed.fit([data,embedding_pos], labels, epochs=50, verbose=0)
    # evaluate the model
    loss, accuracy = model_embed.evaluate(data,embedding_pos,labels, verbose=0)
    print('Accuracy: %f' % (accuracy*100)) 
    
    left_branch = Sequential()
    left_branch.add(Embedding(vocab_size, 50, input_length=23, weights=[embedding_matrix], trainable=False))
    
    right_branch = Sequential()
    right_branch.add(Embedding(38, 38, input_length=23, weights=[embedding_matrix_pos], trainable=False))
    
    merged = Merge([left_branch, right_branch], mode='concat')
    conv1_1 = Conv1D(filters=128, kernel_size=3)(merged)
    model_embed = Sequential()
    model_embed.add(merged)
    model_embed.add(Flatten())
    model_embed.add(Dense(10, activation='softmax'))
    model_embed.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy',   #if i want to change the learning rate of optimizer
                  optimizer=Adam(lr=0.0001, decay=1e-6),
                  metrics=['accuracy'])
    print(model_embed.summary())
    # fit the model
    model_embed.fit([data,embedding_pos], labels, epochs=50, verbose=0)
    # evaluate the model
    loss, accuracy = model_embed.evaluate(data,embedding_pos,labels, verbose=0)
    print('Accuracy: %f' % (accuracy*100))      
    
    print (embedding_dict.get('i'))
    print (embedding_matrix[1])
    if (np.array_equal(embedding_dict.get('i'), embedding_matrix[1])):
        print ('yes')'''
        
    
