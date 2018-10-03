import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir('C:/Users/DELL/Desktop/data saved')
df1 = pd.read_excel('gross.xls')
df1 = df1.ix[4:,'Unnamed: 1':]
df1['Unnamed: 1'][4] = 'year'
df1['Unnamed: 1'][5] = 'quarter'
df1 = df1.set_index('Unnamed: 1')
df1 = df1.T

df2 = pd.read_excel('comp_e.xls')
df2 = df2.ix[4:,'Unnamed: 1':]
df2['Unnamed: 1'][4] = 'year'
df2['Unnamed: 1'][5] = 'quarter'
df2 = df2.set_index('Unnamed: 1')
df2 = df2.T
nam1 = ['GDP','NetDP'] 
names1 = list(df1)
names2 = list(df2)
var1 = {}
var1[nam1[0]] = np.array(df1[names1[3]])
var1[nam1[1]] = np.array(df1[names1[6]])
var1['DEP'] = var1['GDP'] - var1['NetDP'] 
var1['ratio_dep_ndp'] = var1['DEP']/var1['NetDP']




variables = ['Compensation of employees', '        National income',  "Proprietors' income with IVA and CCAdj",
             'Rental income of persons with CCAdj', 'Corporate profits with IVA and CCAdj',
             'Net interest and miscellaneous payments', 'Taxes on production and imports',
             'Less: Subsidies2']
df2 = df2.ix[:,variables]

New_names = ['CE','Y','PI','RI','CP','NI','T','S']
var2 = {}

for idx, name in enumerate(variables):
    var2[New_names[idx]]= np.array(list(df2[name]))
    
var2['DEP'] = var1['DEP'] 


theta = (var2['RI']+var2['CP']+var2['NI']+var2['T']-var2['S'] + var2['DEP'])/(var2['Y']-var2['PI'] + var2['DEP'])
var2['PI_k']=theta*var2['PI']
var2['PI_h']=(1-theta)*var2['PI']
del var2['PI']
var2['LS'] = (var2['CE']+var2['PI_h'])/(var2['Y']+var2['DEP'])
var2['date'] = np.array(list(df1['year']))

f, ax = plt.subplots(1,1)
f.set_figheight(5)
f.set_figwidth(10)
ax.plot(var2['date'], var2['LS'])
ax.set_xlabel('date')
ax.set_ylabel('labor share')
ax.set_title('Labor share in the US from 1948-2017, (gross)')




df3 = pd.read_excel('sect.xls')
df3 = df3.ix[4:,'Unnamed: 1':]
df3['Unnamed: 1'][4] = 'year'
df3['Unnamed: 1'][5] = 'quarter'
df3 = df3.set_index('Unnamed: 1')
df3 = df3.T
names3 = list(df3)
df3 = df3.ix[:,:names3[11]]
names3 = list(df3)
names3 = [names3[i] for i in [4,5,8,9,10]]
new_name3 = ['NetDCorpI','CE','CP','NETi','T_S']
var3 = {}
for idx, name in enumerate(names3):
    var3[new_name3[idx]]=np.array(list(df3[name])) 
    
var3['ratio_dep_ndp']=var1['ratio_dep_ndp'][1:]
var3['GDP'] = (1+var3['ratio_dep_ndp'])*var3['NetDCorpI']

var3['LS'] =  var3['CE']/var3['GDP']
var3['date'] = var2['date'][1:]

f2, ax2 = plt.subplots(1,1)
f2.set_figheight(5)
f2.set_figwidth(10)
ax2.plot(var3['date'], var3['LS'])
ax2.set_xlabel('date')
ax2.set_ylabel('labor share')
ax2.set_title('Labor share in US for the corporate sector from 1948-2017 (gross)')





    
