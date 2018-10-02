import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mp
import os
os.chdir('C:/Users/DELL/Desktop/Quant_macro/Pset2')
df = pd.read_excel('corp_spain.xlsx',header=0)
df = df.ix[:,:'2009']
df = df.set_index('Unnamed: 0')
df=df.T
var = {}
names = list(df)
for name in names:
    var[name] = np.array(list(df[name]))

var['LS'] = var[names[0]]/var[names[1]]

date = [np.float(date) for date in df.index]
f, ax = plt.subplots(1,1)
f.set_figheight(5)
f.set_figwidth(10)
ax.plot(date,var['LS'])
ax.set_xlabel('date')
ax.set_ylabel('labor share')
ax.set_title('Labor share in Spain for the corporate sector from 1980-2009')