import numpy as np
from numpy import vectorize

@vectorize
def U(c, h, kappa, nu):
    if c<=0:
        u = -np.inf
    elif c>0:
        u = np.log(c) - (kappa*h**(1+1/nu))/((1+1/nu))
    return u


class rep_agent:
    def __init__(self, theta, beta, delta, kappa, nu, h=1):
        self.theta = theta
        self.beta = beta
        self.delta = delta
        self.kappa = kappa
        self.nu = nu
        self.h = h
        
    def update(self, V, K, Kp, ret = 1):
        n, c = K.shape
        Vold = np.tile(V, (c,1))# think about it
        C = (1-self.delta)*K + K**(1-self.theta) *self.h**self.theta - Kp
        X = U(C, 1, self.kappa, self.nu) + self.beta*Vold
        argm_pos = np.argmax(X, axis=1)
        Vnew = []
        gk = []
        gc = []
        for i, idx in enumerate(list(argm_pos)):
            v = X[i,idx]
            g = Kp[i,idx]
            g2 = C[i,idx] 
            Vnew.append(v)
            gk.append(g)
            gc.append(g2)
        Vnew = np.array(Vnew)
        gk = np.array(gk)
        if ret == 1:
            return Vnew
        else:
            return Vnew, gk, gc
        
    def update_mono(self, V, K, Kp, ret = 1):
        n, c = K.shape
        Vold = np.tile(V, (c,1))# think about it
        C = (1-self.delta)*K + K**(1-self.theta) *self.h**self.theta - Kp
        X = U(C, 1, self.kappa, self.nu) + self.beta*Vold
        argm_pos = []
        for i in range(len(X)): # this step ensures monotonicity of the decision rule I only look to the upper triangular elements
            pos = np.argmax(X[i,i:])+i
            argm_pos.append(pos)
            
        Vnew = []
        gk = []
        gc = []
        for i, idx in enumerate(list(argm_pos)):
            v = X[i,idx]
            g = Kp[i,idx]
            g2 = C[i,idx] 
            Vnew.append(v)
            gk.append(g)
            gc.append(g2)
        Vnew = np.array(Vnew)
        gk = np.array(gk)
        if ret == 1:
            return Vnew
        else:
            return Vnew, gk, gc
    
    def update_search(self, V, K, Kp, Oldpos, ret = 1):
        n, c = K.shape
        Vold = np.tile(V, (c,1))# think about it
        C = (1-self.delta)*K + K**(1-self.theta) *self.h**self.theta - Kp
        X = U(C, 1, self.kappa, self.nu) + self.beta*Vold
        argm_pos = []
        for i in range(len(X)): # this step ensures monotonicity of the decision rule I only look to the upper triangular elements
            Sn = int(0.3*(c-Oldpos[i]) + 1)
            pos = np.argmax(X[i,Oldpos[i]:Oldpos[i]+Sn])+Oldpos[i]
            comp_pos = Oldpos[i]+Sn
            j = 1
            while pos == comp_pos and pos != c: # here as long a we reach the upper_bound we keep search
                j=j+1
                comp_pos = comp_pos = Oldpos[i]+j*Sn
                pos = np.argmax(X[i,Oldpos[i]+Sn-1:comp_pos])+Oldpos[i]+Sn-1
            argm_pos.append(pos)
                       
        Vnew = []
        gk = []
        gc = []
        for i, idx in enumerate(list(argm_pos)):
            v = X[i,idx]
            g = Kp[i,idx]
            g2 = C[i,idx] 
            Vnew.append(v)
            gk.append(g)
            gc.append(g2)
        Vnew = np.array(Vnew)
        gk = np.array(gk)
        if ret == 1:
            return Vnew, argm_pos
        else:
            return Vnew, gk, gc
        
    def update_howard(self, V, K, Kp, Oldpos, iteration, step = 20, ret = 1):
        n, c = K.shape
        Vold = np.tile(V, (c,1))# think about it
        C = (1-self.delta)*K + K**(1-self.theta) *self.h**self.theta - Kp
        X = U(C, 1, self.kappa, self.nu) + self.beta*Vold
        argm_pos = []
        if iteration%step==0:
            for i in range(len(X)): # this step ensures monotonicity of the decision rule I only look to the upper triangular elements
                pos = np.argmax(X[i,:])
                argm_pos.append(pos)
        else:
            argm_pos = Oldpos       
                       
        Vnew = []
        gk = []
        gc = []
        for i, idx in enumerate(list(argm_pos)):
            v = X[i,idx]
            g = Kp[i,idx]
            g2 = C[i,idx] 
            Vnew.append(v)
            gk.append(g)
            gc.append(g2)
        Vnew = np.array(Vnew)
        gk = np.array(gk)
        if ret == 1:
            return Vnew, argm_pos
        else:
            return Vnew, gk, gc
    
    def update_conc(self, V, K, Kp, ret = 1):
        n, c = K.shape
        Vold = np.tile(V, (c,1))# think about it
        C = (1-self.delta)*K + K**(1-self.theta) *self.h**self.theta - Kp
        X = U(C, 1, self.kappa, self.nu) + self.beta*Vold
        argm_pos = []
        for i in range(len(X)):
            xold = X[i,0]
            for j in range(c):
                if xold >X[i,j]:
                    pos = [i,int(j-1)]
                    break
                elif j==(c-1):
                    pos = [i,(c-1)]
            argm_pos.append(pos)
                       
        Vnew = []
        gk = []
        gc = []
        for i, idx in enumerate(argm_pos):
            v = X[i,idx[1]]
            g = Kp[i,idx[1]]
            g2 = C[i,idx[1]] 
            Vnew.append(v)
            gk.append(g)
            gc.append(g2)
        Vnew = np.array(Vnew)
        gk = np.array(gk)
        if ret == 1:
            return Vnew
        else:
            return Vnew, gk, gc
    
    def problem(self, Vstart, K, Kp, Tol = 10**(-6), max_iter = 1000, type_val = 'simple'):
        n, c = K.shape
        Vold = Vstart
        Oldpos = np.arange(0,c) # start pos is this one because of monotonicity
        err = 1
        j = 0
        while err>Tol:
            if j%100==0:
                print('iteration:',j)
                print('error:',err)
            if type_val == 'simple':
                Vnew = self.update(Vold, K, Kp)
            elif type_val == 'mono':
                Vnew = self.update_mono(Vold, K, Kp)
            elif type_val == 'search':
                Vnew,  New_pos = self.update_search(Vold, K, Kp, Oldpos)
                Oldpos = New_pos
            elif type_val == 'howard':
                Vnew,  New_pos = self.update_howard(Vold, K, Kp, Oldpos, j)
                Oldpos = New_pos
            elif type_val == 'conc':
                Vnew = self.update_conc(Vold, K, Kp)
                
            err = np.max(np.abs(Vnew-Vold))
            Vold = Vnew
            j = j+1
            if j >= max_iter:
                break
        V, gk, gc = self.update(Vold, K, Kp, ret = 0)
        return V, gk, gc
        
        
    
    
    
    
    
    
    
    
    
    