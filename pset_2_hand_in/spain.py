import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mp
import os
os.chdir('C:/Users/DELL/Desktop/Quant_macro/Pset2')
df = pd.read_excel('spain2.xlsx',header=0)
df = df.set_index('Date')
df=df.T
names = list(df)
names_new = names[:]
names_new[2]='GOGM'
var = {}
for idx, name in enumerate(names):
    var[names_new[idx]] = np.array(df[name]) 
    
var['LS'] = var['CE']/var['GDP']

date = [np.float(date) for date in df.index]
f, ax = plt.subplots(1,1)
f.set_figheight(5)
f.set_figwidth(10)
ax.plot(date,var['LS'])
ax.set_xlabel('date')
ax.set_ylabel('labor share')
ax.set_title('Labor share in Spain from 1970-2017')
#ax = scatter(date,var['LS'])
