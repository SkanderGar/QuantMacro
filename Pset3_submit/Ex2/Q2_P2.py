import numpy as np
import os
os.chdir('C:/Users/DELL/Desktop/Quant_macro/PSET3')
import Agent2 as A
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

grid_a = np.linspace(a_min, a_max, n_i)
grid_h = np.linspace(h_min, h_max, n_i)
Ma = np.tile(grid_a,(n_i,1)).T
Mh = np.tile(grid_h,(n_i,1)).T
Mhp = Mh.T 

r_int = np.linspace(-0.1, 0.5,10)

@jit
def get_GE(Agents,Ma,Mh,Mhp):
    maximum = []
    maxi_pos = []
    h_ex_p = []
    for agent in Agents:
        maxi, max_idx, h_ex_post = agent.find(Ma,Mh,Mhp)
        maximum.append(maxi)
        maxi_pos.append(max_idx)
        h_ex_p.append(h_ex_post)
    return maximum, maxi_pos, h_ex_p

Tol = 1
j = 1
store_m_c = []
for r in r_int:
    print('Iteration:',j)
    Agents = [A.Agent(state['y'][i], state['eta'][i], para['eps_y'], r) for i in range(len(state['eta']))]
    maximum, maxi_pos, h_ex_p = get_GE(Agents,Ma,Mh,Mhp)
    asset = []
    for pos in maxi_pos:
        a = grid_a[pos[0]]
        asset.append(a)   
    res = asset@state['dist']
    store_m_c.append([r,res])
    print(res)
    if np.abs(res) < Tol:
        Tol = np.abs(res)
        store_a = asset
        store_pos = maxi_pos
        store_r = r
        store_res = res
        store_h_ex_p = h_ex_p
        store_V = maximum
    j = j+1

store_m_c = np.array(store_m_c)

store_h = []
store_h_p = []
store_h_p_bad = []
store_h_p_good = []
for idx, pos in enumerate(store_pos):
    store_h.append(grid_h[pos[0]])
    store_h_p.append(grid_h[pos[1]])
    store_h_p_bad.append(grid_h[store_h_ex_p[idx][0]])
    store_h_p_good.append(grid_h[store_h_ex_p[idx][1]])

store_a = np.array(store_a)
store_h = np.array(store_h)
store_h_p = np.array(store_h_p)
store_h_p_bad = np.array(store_h_p_bad)
store_h_p_good = np.array(store_h_p_good)

store_c1 = (1-para['tho'])*state['eta']*store_h + state['y'] + para['T1'] - store_a
store_c2_pos = (1-para['tho'])*(state['eta']+para['eps_y'])*store_h_p + (1+store_r)*store_a + para['T2']
store_c2_neg = (1-para['tho'])*(state['eta']-para['eps_y'])*store_h_p + (1+store_r)*store_a + para['T2']
store_S_r = store_a/((state['y']+state['eta'])*store_h*(1-para['tho']))
store_ls = store_h*state['eta']*(1-para['tho'])/(store_h*state['eta']*(1-para['tho'])+state['y']+para['T1'])
store_ls_fu_pos = store_h_p_good*(state['eta']+para['eps_y'])*(1-para['tho'])/(store_h_p_good*(state['eta']+para['eps_y'])*(1-para['tho'])+(1+store_r)*store_a+para['T2'])
store_ls_fu_neg = store_h_p_bad*(state['eta']-para['eps_y'])*(1-para['tho'])/(store_h_p_bad*(state['eta']-para['eps_y'])*(1-para['tho'])+(1+store_r)*store_a+para['T2'])
store_E_gc = (0.5*store_c2_pos + store_c2_neg*0.5 - store_c1)/store_c1
store_E_gwh = (0.5*store_h_p_good*(state['eta']+para['eps_y'])+0.5*store_h_p_bad*(state['eta']-para['eps_y'])-state['eta']*store_h)/state['eta']*store_h
store_elas = store_E_gc/store_E_gwh
gc_pos = (store_c2_pos-store_c1)/store_c1
gc_neg = (store_c2_neg-store_c1)/store_c1
gwh_pos = (store_h_p_good*(state['eta']+para['eps_y'])-store_h*state['eta'])/store_h*state['eta']
gwh_neg = (store_h_p_bad*(state['eta']+para['eps_y'])-store_h*state['eta'])/store_h*state['eta']
elas_ratio_pos = (gc_pos/gwh_pos)/(store_E_gc/store_E_gwh)
elas_ratio_neg = (gc_neg/gwh_neg)/(store_E_gc/store_E_gwh)

data = np.vstack((store_c1, store_c2_pos, store_c2_neg, store_a, store_h, store_h_p,
                  state['dist'], state['y'], store_S_r, store_ls, store_h_p_bad,
                  store_h_p_good, store_ls_fu_pos, store_ls_fu_neg, store_E_gc,
                  store_E_gwh, store_elas, elas_ratio_pos, elas_ratio_neg, state['eta']*store_h, store_V)).T
                  
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








    
    
    
    
    
    
    