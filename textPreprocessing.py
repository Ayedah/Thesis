# -*- coding: utf-8 -*-
"""
Created on Thu May  3 14:35:19 2018

@author: aidaz

Clean the datasets and save in another csv file
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
import string
import unidecode
#cols = ['sentiment','id','date','query_string','user','text']
df = pd.read_csv("separatedShuffledProcessed_70-30_Testset2.csv",header=0,sep=None)
# above line will be different depending on where you saved your data, and your file name
df.head()
#print (df.text.value_counts())
'''df.drop(['id','date','query_string','user'],axis=1,inplace=True)
df[df.sentiment == 0].head(10)
df['pre_clean_len'] = [len(t) for t in df.text]
print (df.pre_clean_len)


data_dict = {
    'sentiment':{
        'type':df.sentiment.dtype,
        'description':'sentiment class - 0:negative, 1:positive'
    },
    'text':{
        'type':df.text.dtype,
        'description':'tweet text'
    },
    'pre_clean_len':{
        'type':df.pre_clean_len.dtype,
        'description':'Length of the tweet before cleaning'
    },
    'dataset_shape':df.shape
}
pprint(data_dict)'''

cList = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have",
  
}

c_re = re.compile('(%s)' % '|'.join(cList.keys()))

def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)

tok = WordPunctTokenizer()
#pat1 = r'@[A-Za-z0-9]+'
#pat2 = r'https?://[A-Za-z0-9./]+'
#combined_pat = r'|'.join((pat1, pat2))
def tweet_cleaner(text):
    consequitivedots = re.compile(r'\.{3,}')
    res=consequitivedots.sub('', text)
    #res=consequitivedots.sub(, text)
    #soup = BeautifulSoup(text, 'lxml')
    #souped = soup.get_text()
    #stripped = re.sub(combined_pat, '', souped)
    #try:
        #clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    #except:
        #clean = stripped
    #letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = res.lower()
    #print(lower_case)
    sentence = lower_case.strip()
    new_sentence=expandContractions(sentence)
    remove = string.punctuation
    remove = remove.replace(".", "") 
    remove = remove.replace("?", "")
    remove = remove.replace("!", "")
    remove = remove.replace(",", "")
    remove = remove.replace("-", "")
    remove = remove.replace("#", "")
    remove = remove.replace("+", "")
    remove = remove.replace("_", "")
    remove = remove.replace("/", "")
    #pattern = r"[{}]".format(remove)
    filtered_sentence=new_sentence.translate({ord(char): None for char in remove})
    filtered_sentence = ''.join(i for i in filtered_sentence if not i.isdigit())
    #PATTERN = r'[^\P{P}!|.|,|?]+'
    #+ / _ - #
    #filtered_sentence = re.sub(PATTERN, r' ', new_sentence)
    #print(filtered_sentence)
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    #print(new_sentence)
    print (filtered_sentence)
    res = unidecode.unidecode(filtered_sentence)
    #stop_words = set(stopwords.words('english'))
    words = tok.tokenize(text)
    '''filtered_sentence = []
 
    for w in words:
        if w not in stop_words:
            filtered_sentence.append(w)
    #print(words)'''
    tokens = [token.strip() for token in words]
    #print(tokens)
    
    return (" ".join(tokens)).strip()
#tweet_cleaner('Testing.... how IS it working!')
ranges=df.text.count()
nums = [0,ranges]
print ("Cleaning and parsing the tweets...\n")
clean_SE_texts = []
for i in range(nums[0],nums[1]):
    if( (i+1)%10000 == 0 ):
        print ("Tweets %d of %d has been processed" % ( i+1, nums[1] )  )                                                                  
    clean_SE_texts.append(tweet_cleaner(df['text'][i]))
    
clean_df = pd.DataFrame(clean_SE_texts,columns=['text'])
print(clean_df.count())
#clean_df=clean_df.drop_duplicates(subset='text', keep='first')
print(clean_df.count())
clean_df['labels'] = df['labels']
clean_df= clean_df.dropna(subset=['text'])
print(clean_df.count())
clean_df.head()
clean_df.to_csv('NewTest.csv',encoding='utf-8')
#csv = 'separatedProcessed_70-30.csv'
#my_df = pd.read_csv(csv,index_col=0)
#my_df.head()





