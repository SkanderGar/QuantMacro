import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir('C:/Users/DELL/Desktop/Quant_macro/Pset2')
df = pd.read_excel('download_all.xls')
date = df.ix[4,'Unnamed: 1':]
dates = [np.float(da) for da in date[1:]]

dat = []
dat.append(dates[0])
for idx, dats in enumerate(dates):
    mod = (idx+1)%4
    if mod == 1:
        data = dats
    else:
        data = dat[-1]+0.25
    dat.append(data)

del dat[0]


df = df.ix[6:,'Unnamed: 1':]
df = df.set_index('Unnamed: 1')
variables = ['Compensation of employees', '        National income',  "Proprietors' income with IVA and CCAdj",
             'Rental income of persons with CCAdj', 'Corporate profits with IVA and CCAdj',
             'Net interest and miscellaneous payments', 'Taxes on production and imports',
             'Less: Subsidies2']
df = df.ix[variables,:]
df = df.T
df_np = np.array(df)

New_names = ['CE','Y','PI','RI','CP','NI','T','S']

data_l = {}
for idx, var in enumerate(variables):
    col = np.array(list(df[var]))
    data_l[New_names[idx]] = col

theta = (data_l['RI']+data_l['CP']+data_l['NI']+data_l['T']-data_l['S'])/(data_l['Y']-data_l['PI'])
data_l['PI_k']=theta*data_l['PI']
data_l['PI_h']=(1-theta)*data_l['PI']
del data_l['PI']
data_l['LS'] = (data_l['CE']+data_l['PI_h'])/data_l['Y']
data_l['date'] = dat
f, ax = plt.subplots(1,1)
f.set_figheight(5)
f.set_figwidth(10)
ax.plot(data_l['date'], data_l['LS'])
ax.set_xlabel('date')
ax.set_ylabel('labor share')
ax.set_title('Labor share in the US from 1948-2017')










