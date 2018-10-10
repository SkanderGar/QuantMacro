import numpy as np
from numpy import vectorize

@vectorize
def U(c):
    if c <= 0:
        u = -np.inf
    else:
        u = np.log(c)
    return u

class model1:
    def __init__(self, z, h=0.31, beta=0.98, theta=0.67, delta=0.0625):
        self.z = z
        self.h = h
        self.beta = beta
        self.theta = theta
        self.delta = delta
        
    def update(self, V, Km, Kpm, ask=0):
        Vm = np.tile(V,(len(Km),1))
        C = Km**(1-self.theta) *(self.h*self.z)**self.theta + (1-self.delta)*Km - Kpm
        Vnew = U(C) + self.beta*Vm
        m_idx = np.argmax(Vnew,axis=1)
        Vnew_f = []
        g_c = []
        g_k = []
        for i in range(len(Vnew)):
            Vnew_f.append(Vnew[i,m_idx[i]])
            g_c.append(C[i,m_idx[i]])
            g_k.append(Kpm[i,m_idx[i]])
        Vnew_f = np.array(Vnew_f)
        g_c = np.array(g_c)
        g_k = np.array(g_k)
        
        if ask == 0:
            return Vnew_f
        elif ask == 1:
            return Vnew_f, g_c, g_k
    
    def problem(self, Vstart, Km, Kpm, Tol = 10**(-3)): #the tolerance is in percent
        err = 1
        Vold = Vstart
        j = 0
        while err > Tol:
            Vnew = self.update(Vold, Km, Kpm)
            err = np.max(np.abs((Vnew-Vold)/Vold))
            Vold = Vnew
            j += 1
            print(j)
        V, g_c, g_k =  self.update(Vold, Km, Kpm, ask=1)
        return V, g_c, g_k
    
    def trans(self, Vss1, g_kss1, Km, Kpm, Tol = 10**(-3)): #the tolerance is in percent
        gridk = Km[:,0]
        f_g_kss1 = lambda x: np.interp(x, gridk, g_kss1)
        f_Vss1 = lambda x: np.interp(x, gridk, Vss1)
        Kf = []
        Vf = []
        Kf.append(f_g_kss1)
        Vf.append(f_Vss1)
        err = 1
        gold = g_kss1
        Vold = Vss1
        j = 0
        while err > Tol:
            Vnew, g_c, g_k= self.update(Vold, Km, Kpm, ask=1)
            gnew = g_c
            f_g_kt = lambda x: np.interp(x, gridk, g_k)
            f_Vt = lambda x: np.interp(x, gridk, Vnew)
            Kf.append(f_g_kt)
            Vf.append(f_Vt)
            err = np.max(np.abs((gnew-gold)/gold))
            gold = gnew
            Vold = Vnew
            j += 1
            print(j)
        return Vf, Kf
    
    
