# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 20:50:27 2018

@author: aidaz
"""
import pandas as pd
import numpy as np
df=pd.read_csv('separatedProcessed_70-30Shortened.csv',header=0,sep=None)
data=df
data1=data.iloc[0:607373,:]
data2=data.iloc[607373:729550,:]
data3=data.iloc[729550:878976,:]

clean_df = pd.DataFrame(data1)
clean_df = clean_df.sample(frac=1).reset_index(drop=True)
clean_df1 = pd.DataFrame(data2)
clean_df1 = clean_df1.sample(frac=1).reset_index(drop=True)
clean_df2 = pd.DataFrame(data3)
clean_df2 = clean_df2.sample(frac=1).reset_index(drop=True)

dt=[clean_df,clean_df1,clean_df2]

clean_df=pd.concat(dt,ignore_index=True)
clean_df.head()
print(clean_df.text.count())
#clean_df.to_csv('Book3_70-30.csv',encoding='utf-8',index=True)
clean_df.to_csv('separatedShuffledProcessed_70-30.csv.gz',compression='gzip')