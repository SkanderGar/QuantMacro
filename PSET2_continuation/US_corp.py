import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir('C:/Users/DELL/Desktop/data saved')
df1 = pd.read_excel('sect.xls')
df1 = df1.ix[4:,'Unnamed: 1':]
df1['Unnamed: 1'][4] = 'year'
df1['Unnamed: 1'][5] = 'quarter'
df1 = df1.set_index('Unnamed: 1')
df1 = df1.T
names = list(df1)
df1 = df1.ix[:,:names[11]]
names = list(df1)
names = [names[i] for i in [4,5,8,9,10]]
new_name = ['NetDCorpI','CE','CP','NETi','T_S']
var = {}
for idx, name in enumerate(names):
   var[new_name[idx]]=np.array(list(df1[name])) 
