import numpy as np
import matplotlib.pyplot as plt

market = np.loadtxt('Market.txt')
data1 = np.loadtxt('variables.txt')
data_e1 = data1[0:100,:]
data_e2 = data1[100:200,:]
data_e3 = data1[200:300,:]
data_e4 = data1[300:400,:]

count = 0
idx = []
for i in range(len(data_e1)):
    if data_e1[i,7] == 0.001:
        count = count+1
        idx.append(i)
a = np.arange(0,idx[1])
b = np.arange(idx[-1]+1,len(data_e1))

new_e1 = np.vstack((data_e1[a,:],data_e1[b,:]))
new_e1[0,6] = new_e1[0,6]*count # changing the distribution

new_e2 = np.vstack((data_e2[a,:],data_e2[b,:]))
new_e2[0,6] = new_e2[0,6]*count 

new_e3 = np.vstack((data_e3[a,:],data_e3[b,:]))
new_e3[0,6] = new_e3[0,6]*count 

new_e4 = np.vstack((data_e4[a,:],data_e4[b,:]))
new_e4[0,6] = new_e4[0,6]*count 

data1 = None
data_e1 = None
data_e2 = None
data_e3 = None
data_e4 = None

f, (ax1,ax2) = plt.subplots(1,2)
f.set_figheight(5)
f.set_figwidth(10)


ax1.plot(new_e1[:,7],new_e1[:,0],color='b', label = 'eta_y=1')
ax1.plot(new_e2[:,7],new_e2[:,0],color='r', label = 'eta_y=1.5')
ax1.plot(new_e3[:,7],new_e3[:,0],color='g', label = 'eta_y=2.5')
ax1.plot(new_e4[:,7],new_e4[:,0],color='k', label = 'eta_y=3')
ax1.set_xlabel('Y_0')
ax1.set_ylabel('C')
ax1.set_ylim(0.5,1.8)
ax1.legend(loc = 'upper right')
ax1.set_title('Consumption as function of Y_0')


ax2.plot(new_e1[:,7],new_e1[:,3],color='b', label = 'eta_y=1')
ax2.plot(new_e2[:,7],new_e2[:,3],color='r', label = 'eta_y=1.5')
ax2.plot(new_e3[:,7],new_e3[:,3],color='g', label = 'eta_y=2.5')
ax2.plot(new_e4[:,7],new_e4[:,3],color='k', label = 'eta_y=3')
ax2.set_xlabel('Y_0')
ax2.set_ylabel('a')
ax2.legend(loc = 'upper right')
ax2.set_title('Assets as function of Y_0')



f2, (ax3,ax4) = plt.subplots(1,2)
f2.set_figheight(5)
f2.set_figwidth(10)


ax3.plot(new_e1[:,7],new_e1[:,1],color='b', label = 'eta_y=1')
ax3.plot(new_e2[:,7],new_e2[:,1],color='r', label = 'eta_y=1.5')
ax3.plot(new_e3[:,7],new_e3[:,1],color='g', label = 'eta_y=2.5')
ax3.plot(new_e4[:,7],new_e4[:,1],color='k', label = 'eta_y=3')
ax3.set_xlabel('Y_0')
ax3.set_ylabel('C+')
ax3.legend(loc = 'upper right')
ax3.set_title('Future consumption with positive shock')

ax4.plot(new_e1[:,7],new_e1[:,2],color='b', label = 'eta_y=1')
ax4.plot(new_e2[:,7],new_e2[:,2],color='r', label = 'eta_y=1.5')
ax4.plot(new_e3[:,7],new_e3[:,2],color='g', label = 'eta_y=2.5')
ax4.plot(new_e4[:,7],new_e4[:,2],color='k', label = 'eta_y=3')
ax4.set_xlabel('Y_0')
ax4.set_ylabel('C-')
ax4.legend(loc = 'upper right')
ax4.set_title('Future consumption with negative shock')

f3, ax5 = plt.subplots(1,1)
f3.set_figheight(5)
f3.set_figwidth(10)


ax5.plot(new_e1[:,7],new_e1[:,8],color='b', label = 'eta_y=1')
ax5.plot(new_e2[:,7],new_e2[:,8],color='r', label = 'eta_y=1.5')
ax5.plot(new_e3[:,7],new_e3[:,8],color='g', label = 'eta_y=2.5')
ax5.plot(new_e4[:,7],new_e4[:,8],color='k', label = 'eta_y=3')
ax5.set_xlabel('Y_0')
ax5.set_ylabel('Saving rate')
ax5.legend(loc = 'upper right')
ax5.set_title('Saving rate given Y_0')


f4, (ax6,ax7,ax8) = plt.subplots(1,3)
f4.set_figheight(5)
f4.set_figwidth(12)


ax6.plot(new_e1[:,7],new_e1[:,4],color='b', label = 'eta_y=1')
ax6.plot(new_e2[:,7],new_e2[:,4],color='r', label = 'eta_y=1.5')
ax6.plot(new_e3[:,7],new_e3[:,4],color='g', label = 'eta_y=2.5')
ax6.plot(new_e4[:,7],new_e4[:,4],color='k', label = 'eta_y=3')
ax6.set_xlabel('Y_0')
ax6.set_ylabel('h')
ax6.legend(loc = 'upper right')
ax6.set_title('labor supply in the first period')

ax7.plot(new_e1[:,7],new_e1[:,10],color='b', label = 'eta_y=1')
ax7.plot(new_e2[:,7],new_e2[:,10],color='r', label = 'eta_y=1.5')
ax7.plot(new_e3[:,7],new_e3[:,10],color='g', label = 'eta_y=2.5')
ax7.plot(new_e4[:,7],new_e4[:,10],color='k', label = 'eta_y=3')
ax7.set_xlabel('Y_0')
ax7.set_ylabel('h-')
ax7.legend(loc = 'upper right')
ax7.set_title('labor supply good shock')

ax8.plot(new_e1[:,7],new_e1[:,11],color='b', label = 'eta_y=1')
ax8.plot(new_e2[:,7],new_e2[:,11],color='r', label = 'eta_y=1.5')
ax8.plot(new_e3[:,7],new_e3[:,11],color='g', label = 'eta_y=2.5')
ax8.plot(new_e4[:,7],new_e4[:,11],color='k', label = 'eta_y=3')
ax8.set_xlabel('Y_0')
ax8.set_ylabel('h+')
ax8.legend(loc = 'upper right')
ax8.set_title('labor supply bad shock')



h1, (axh1,axh2) = plt.subplots(1,2)
h1.set_figheight(5)
h1.set_figwidth(10)


axh1.plot(new_e1[:,7],new_e1[:,19],color='b', label = 'eta_y=1')
axh1.plot(new_e2[:,7],new_e2[:,19],color='r', label = 'eta_y=1.5')
axh1.plot(new_e3[:,7],new_e3[:,19],color='g', label = 'eta_y=2.5')
axh1.plot(new_e4[:,7],new_e4[:,19],color='k', label = 'eta_y=3')
axh1.set_xlabel('Y_0')
axh1.set_ylabel('wh')
axh1.legend(loc = 'upper right')
axh1.set_title('labor income given Y_0')

axh2.plot(new_e1[:,7],new_e1[:,9],color='b', label = 'eta_y=1')
axh2.plot(new_e2[:,7],new_e2[:,9],color='r', label = 'eta_y=1.5')
axh2.plot(new_e3[:,7],new_e3[:,9],color='g', label = 'eta_y=2.5')
axh2.plot(new_e4[:,7],new_e4[:,9],color='k', label = 'eta_y=3')
axh2.set_xlabel('Y_0')
axh2.set_ylabel('ls')
axh2.legend(loc = 'upper right')
axh2.set_title('After tax labor share')


h2, (axh3,axh4) = plt.subplots(1,2)
h2.set_figheight(5)
h2.set_figwidth(10)


axh3.plot(new_e1[:,7],new_e1[:,12],color='b', label = 'eta_y=1')
axh3.plot(new_e2[:,7],new_e2[:,12],color='r', label = 'eta_y=1.5')
axh3.plot(new_e3[:,7],new_e3[:,12],color='g', label = 'eta_y=2.5')
axh3.plot(new_e4[:,7],new_e4[:,12],color='k', label = 'eta_y=3')
axh3.set_xlabel('Y_0')
axh3.set_ylabel('ls+')
axh3.legend(loc = 'upper right')
axh3.set_title('labor share positif shock given Y_0')

axh4.plot(new_e1[:,7],new_e1[:,13],color='b', label = 'eta_y=1')
axh4.plot(new_e2[:,7],new_e2[:,13],color='r', label = 'eta_y=1.5')
axh4.plot(new_e3[:,7],new_e3[:,13],color='g', label = 'eta_y=2.5')
axh4.plot(new_e4[:,7],new_e4[:,13],color='k', label = 'eta_y=3')
axh4.set_xlabel('Y_0')
axh4.set_ylabel('ls-')
axh4.legend(loc = 'upper right')
axh4.set_title('laboe share negative shock given Y_0')


h3, (axh5,axh6) = plt.subplots(1,2)
h3.set_figheight(5)
h3.set_figwidth(10)


axh5.plot(new_e1[:,7],new_e1[:,15],color='b', label = 'eta_y=1')
axh5.plot(new_e2[:,7],new_e2[:,15],color='r', label = 'eta_y=1.5')
axh5.plot(new_e3[:,7],new_e3[:,15],color='g', label = 'eta_y=2.5')
axh5.plot(new_e4[:,7],new_e4[:,15],color='k', label = 'eta_y=3')
axh5.set_xlabel('Y_0')
axh5.set_ylabel('Egc')
axh5.legend(loc = 'upper right')
axh5.set_title('Expected consumption growth')

axh6.plot(new_e1[:,7],new_e1[:,16],color='b', label = 'eta_y=1')
axh6.plot(new_e2[:,7],new_e2[:,16],color='r', label = 'eta_y=1.5')
axh6.plot(new_e3[:,7],new_e3[:,16],color='g', label = 'eta_y=2.5')
axh6.plot(new_e4[:,7],new_e4[:,16],color='k', label = 'eta_y=3')
axh6.set_xlabel('Y_0')
axh6.set_ylabel('Egwh')
axh6.legend(loc = 'upper right')
axh6.set_title('Expected income growth')



h4, (axh7,axh8) = plt.subplots(1,2)
h4.set_figheight(5)
h4.set_figwidth(10)


axh7.plot(new_e1[:,7],new_e1[:,17],color='b', label = 'eta_y=1')
axh7.plot(new_e2[:,7],new_e2[:,17],color='r', label = 'eta_y=1.5')
axh7.plot(new_e3[:,7],new_e3[:,17],color='g', label = 'eta_y=2.5')
axh7.plot(new_e4[:,7],new_e4[:,17],color='k', label = 'eta_y=3')
axh7.set_xlabel('Y_0')
axh7.set_ylabel('(gc/gwh)/(Egc/Egwh)+')
axh7.legend(loc = 'upper right')
axh7.set_title('elasticity vs expected elasticity')

axh8.plot(new_e1[:,7],new_e1[:,18],color='b', label = 'eta_y=1')
axh8.plot(new_e2[:,7],new_e2[:,18],color='r', label = 'eta_y=1.5')
axh8.plot(new_e3[:,7],new_e3[:,18],color='g', label = 'eta_y=2.5')
axh8.plot(new_e4[:,7],new_e4[:,18],color='k', label = 'eta_y=3')
axh8.set_xlabel('Y_0')
axh8.set_ylabel('(gc/gwh)/(Egc/Egwh)-')
axh8.legend(loc = 'upper right')
axh8.set_title('elasticity vs expected elasticity')


g1, axg1 = plt.subplots(1,1)
g1.set_figheight(5)
g1.set_figwidth(10)

axg1.plot(market[:,0],market[:,1],color='b')

axg1.set_xlabel('r')
axg1.set_ylabel('A')
axg1.set_title('Market excess demand or supply in assets')


g2, axg2 = plt.subplots(1,1)
g2.set_figheight(5)
g2.set_figwidth(10)


axg2.plot(new_e1[:,7],new_e1[:,20],color='b', label = 'eta_y=1')
axg2.plot(new_e2[:,7],new_e2[:,20],color='r', label = 'eta_y=1.5')
axg2.plot(new_e3[:,7],new_e3[:,20],color='g', label = 'eta_y=2.5')
axg2.plot(new_e4[:,7],new_e4[:,20],color='k', label = 'eta_y=3')
axg2.set_xlabel('Y_0')
axg2.set_ylabel('V')
axg2.legend(loc = 'upper right')
axg2.set_title('Life time utility by initial wealth')




