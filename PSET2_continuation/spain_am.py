import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir('C:/Users/DELL/Desktop/data saved')
df1 = pd.read_excel('AMECO.xlsx')
del df1['Unnamed: 1']
df1['AMECO RESULTS'][2]='date'
df1 = df1.ix[2:,'AMECO RESULTS':]
df1 = df1.set_index('AMECO RESULTS')
df1 = df1.T
names = list(df1)
new_names = ['date','GOS','NOS','GOS_adj_CE','NOS_adj_CE','T_S','GNI','CE' ]

var = {}
for idx, name in enumerate(names):
    var[new_names[idx]] = np.array(df1[name])
    
var['PI_h'] = var['GOS'] - var['GOS_adj_CE'] 
var['LS'] = (var['CE']+var['PI_h'])/var['GNI']


LS = np.flip(var['LS'],0)
date = np.flip(var['date'],0)
f, ax = plt.subplots(1,1)
f.set_figheight(5)
f.set_figwidth(10)
ax.plot(date, LS)
ax.set_xlabel('date')
ax.set_ylabel('labor share')
ax.set_title('Labor share Spain from 1960-2018, (gross)')

    

