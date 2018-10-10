import numpy as np
import os
os.chdir('C:/Users/DELL/Desktop/fresh start')
import PS3_agent as A
import matplotlib.pyplot as plt
from numba import jit

################## parameters ##############
i = 400
n=100

n_i = 100
a_max = 1
a_min = -1
h_max = 1
h_min = 0

para = {}
para['beta']=0.99
para['eta_y']=[1, 1.5, 2.5, 3]
para['eps_y']=0.05
para['tho']=0
para['T1'] = 0
para['T2'] = 0
##################

y_0 = np.linspace(0.001, 0.009, 100)
for idx, y in enumerate(y_0):
    if y>=0.0055 and y<=0.0087:
       y_0[idx]=0.001 
y_0 = np.hstack((y_0,y_0,y_0,y_0))
One = np.ones(n)
eta = para['eta_y'][0]*One
for idx in range(1,len(para['eta_y'])):
    eta = np.hstack((eta,para['eta_y'][idx]*One))
    
state = {}
state['y']=y_0
state['eta']=eta
########### compute distribution nuber of people 
state['dist'] = 1/400 *np.ones(i)

#########################
r_int = np.linspace(0, 0.03,10)


def G_eq(Agents):
    a = []
    h = []
    hp = []
    hpos = []
    hneg = []
    for agent in Agents:
        val = agent.problem()
        h1, h2 = agent.problem_expost()
        a.append(val[0])
        h.append(val[1])
        hp.append(val[2])
        hpos.append(h1)
        hneg.append(h2)
    Vals = np.hstack((a,h,hp,hpos,hneg))
    return Vals

Tol = 1
j = 1
store_m_c = []
for r in r_int:
    print('Iteration:',j)
    Agents = [A.Agent(state['y'][i], state['eta'][i], para['eps_y'], r) for i in range(len(state['eta']))]
    a_h_hp = G_eq(Agents)
    res = a_h_hp[:,0]@state['dist']
    store_m_c.append([r,res])
    print(res)
    j = j+1
    if np.abs(res) < Tol:
        True_Ag = Agents
        Tol = np.abs(res)
        store_a = a_h_hp[:,0]
        store_h = a_h_hp[:,1]
        store_hp_exante = a_h_hp[:,2]
        store_hpos = a_h_hp[:,3]
        store_hneg = a_h_hp[:,4]
        store_r = r
        store_res = res
    
W = [agent.wealfare(para['T1'], para['T1']) for agent in True_Ag]  
W = np.array(W)
W.shape = (400,) 
    
    
store_c1 = (1-para['tho'])*state['eta']*store_h + state['y'] + para['T1'] - store_a
store_c2_pos = (1-para['tho'])*(state['eta']+para['eps_y'])*store_hpos + (1+store_r)*store_a + para['T2']
store_c2_neg = (1-para['tho'])*(state['eta']-para['eps_y'])*store_hneg + (1+store_r)*store_a + para['T2']    
store_S_r = store_a/((state['y']+state['eta'])*store_h*(1-para['tho']))
store_ls = store_h*state['eta']*(1-para['tho'])/(store_h*state['eta']*(1-para['tho'])+state['y']+para['T1'])
store_ls_fu_pos = store_hpos*(state['eta']+para['eps_y'])*(1-para['tho'])/(store_hpos*(state['eta']+para['eps_y'])*(1-para['tho'])+(1+store_r)*store_a+para['T2'])
store_ls_fu_neg = store_hneg*(state['eta']-para['eps_y'])*(1-para['tho'])/(store_hneg*(state['eta']-para['eps_y'])*(1-para['tho'])+(1+store_r)*store_a+para['T2'])
store_E_gc = (0.5*store_c2_pos + store_c2_neg*0.5 - store_c1)/store_c1
store_E_gwh = (0.5*store_hpos*(state['eta']+para['eps_y'])+0.5*store_hneg*(state['eta']-para['eps_y'])-state['eta']*store_h)/state['eta']*store_h
store_elas = store_E_gc/store_E_gwh
gc_pos = (store_c2_pos-store_c1)/store_c1
gc_neg = (store_c2_neg-store_c1)/store_c1
gwh_pos = (store_hpos*(state['eta']+para['eps_y'])-store_h*state['eta'])/store_h*state['eta']
gwh_neg = (store_hneg*(state['eta']+para['eps_y'])-store_h*state['eta'])/store_h*state['eta']
elas_ratio_pos = (gc_pos/gwh_pos)/(store_E_gc/store_E_gwh)
elas_ratio_neg = (gc_neg/gwh_neg)/(store_E_gc/store_E_gwh)

data = np.vstack((store_c1, store_c2_pos, store_c2_neg, store_a, store_h, store_hp_exante,
                  state['dist'], state['y'], store_S_r, store_ls, store_hneg,
                  store_hpos, store_ls_fu_pos, store_ls_fu_neg, store_E_gc,
                  store_E_gwh, store_elas, elas_ratio_pos, elas_ratio_neg, state['eta']*store_h, W)).T
                  
final_para = {}
final_para['interest'] = store_r
final_para['tho'] = para['tho']
final_para['T1'] = para['T1']
final_para['T2'] = para['T2']
final_para['residual of assets market'] = store_res
final_para['eta_y'] = para['eta_y']
names = list(final_para)


file = open('parameters.txt','w')
for name in names:
    file.write(f'{name}:{final_para[name]} \n')
file.close()

np.savetxt('variables.txt',data)
np.savetxt('Market.txt',store_m_c)

    
    
    
    
    

