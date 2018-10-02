import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir('C:/Users/DELL/Desktop/Quant_macro/Pset2')
df = pd.read_excel('download_corp.xls')
date = df.ix[4,'Unnamed: 1':]
dates = [np.float(da) for da in date[1:]]
df = df.ix[6:,'Unnamed: 1':]
names = list(df['Unnamed: 1'])
names_needed = [names[i] for i in [2,3,6,7,8]]
df = df.set_index('Unnamed: 1')
df = df.ix['            National income':'    Noncorporate business',:]
df = df.ix[names_needed,:]
df=df.T
new_name = ['CB','CE','CP','NE','T_S']
new_data = {}
for idx, name in enumerate(names_needed):
    new_data[new_name[idx]] = np.array(list(df[name]))


new_data['LS'] =  new_data['CE']/new_data['CB']
new_data['date'] = np.array(dates) 

f, ax = plt.subplots(1,1)
f.set_figheight(5)
f.set_figwidth(10)
ax.plot(new_data['date'], new_data['LS'])
ax.set_xlabel('date')
ax.set_ylabel('labor share')
ax.set_title('Labor share in US for the corporate sector from 1948-2017')
