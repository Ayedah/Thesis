# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 22:39:56 2018

@author: aidaz
"""
import pandas as pd
import numpy as np
df=pd.read_csv('comments-strong-valence.csv.gz',header=0)
#df = pd.read_csv("Book1.csv",header=0)
df.head()
print(df.text.count())
print(df.columns)

df['senti.min.autotime'].replace([-1,-2,-3,-4,-5], [1,2,3,4,5], inplace=True)  #switch polarities as wrongly labeled in data
df['senti.max.autotime'].replace([1,2,3,4,5], [-1,-2,-3,-4,-5], inplace=True)

df.rename(columns={'senti.max.autotime':'senti.min.autotime',
                          'senti.min.autotime':'senti.max.autotime'}, 
                 inplace=True)
print(df.columns)

print(df.head(5))
df=df.drop_duplicates(subset='text', keep='first')
df= df.dropna(subset=['text'])
print(df.text.count())
df['class'] = np.where(df['senti.max.autotime']>=4, 1, 0)      
#clean_df['target'] = df.sentiment
df.drop(['source','bug.id','comment.id','senti.min','senti.max','tensi.max','tensi.min','tensi.max.autotime','tensi.min.autotime'],axis=1,inplace=True)
count=len(df[(df['class']<0)])
print (count)
print(df.text.count()-count)
total=int((count*100)/30)
print(total)
select=int(round(total-count))
print(select)
data1 =df.loc[df['class'].isin([0])].groupby('class').head(count)
data = df.loc[df['class'].isin([1])].groupby('class').head(select)
#X_train, X_test, y_train, y_test = train_test_split(data, class, test_size=0.2)
print (type(data))
#df = pd.DataFrame(np.random.random((total,2)))
#print (df)

a, TestP = np.split(data,[int(round((.8*len(data))))])
print(a)

trainP, validateP = np.split(a,[int(round((.8*len(a))))])

c, TestN = np.split(data1, [int(round((.9*len(data1))))])
print(c)

trainN, validateN = np.split(c,[int(round((.9*len(c))))])
clean_df = pd.DataFrame(trainP)
clean_df = clean_df.sample(frac=1).reset_index(drop=True)
clean_df1 = pd.DataFrame(trainN)
clean_df1 = clean_df1.sample(frac=1).reset_index(drop=True)
clean_df2 = pd.DataFrame(validateP)
clean_df2 = clean_df2.sample(frac=1).reset_index(drop=True)
clean_df3 = pd.DataFrame(validateN)
clean_df3 = clean_df3.sample(frac=1).reset_index(drop=True)
clean_df4 = pd.DataFrame(TestP)
clean_df4= clean_df4.sample(frac=1).reset_index(drop=True)
clean_df5 = pd.DataFrame(TestN)
clean_df5 = clean_df5.sample(frac=1).reset_index(drop=True)
dt=[clean_df,clean_df1,clean_df2,clean_df3,clean_df4,clean_df5]

clean_df=pd.concat(dt,ignore_index=True)
#datajoin=pd.concat(clean_df,clean_df1)
#clean_df = pd.DataFrame(data1)
#clean_df['target'] = df.sentiment
clean_df.head()
print(clean_df.text.count())
#clean_df.to_csv('Book3_70-30.csv',encoding='utf-8',index=True)
clean_df.to_csv('separated.csv.gz',compression='gzip')
