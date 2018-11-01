import numpy as np
from scipy.stats import norm
from numpy import vectorize

@vectorize
def U1(C, C_):
    if C <= 0:
        U = -np.inf
    else:
        U = -(1/2)*(C-C_)**2
    return U

@vectorize
def U2(C, S):
    if C <= 0:
        U = -np.inf
    else:
        U = (C**(1-S) -1)/(1-S)
    return U


class agent1:
    
    def __init__(self, N_a, sig_y=0.5, gamma_y=0.7, T=45, N_s=2, rho = 0.06, Sig = 5, C_ = 1, r = 0.04, U2 = 1, B=1, cert = 0):
        self.cert = cert
        self.T = T
        self.gamma_y = gamma_y
        self.sig_y = sig_y
        self.beta = 1/(1+rho)
        self.Sig = Sig
        self.C_ = C_
        self.r = r
        self.U2 = 1
        self.N_s = N_s
        self.N_a = N_a
        self.Tr, self.Y_grid = self.markov_Tr(self.N_s, Sig_y = self.sig_y, gamma = self.gamma_y)
        max_a = self.Y_grid[-1]/self.r
        if B==1:
            min_a = -(self.Y_grid[0]/self.r)*0.98
        else:
            min_a = 0
        self.grid_a = np.linspace(min_a, max_a, self.N_a)
        if self.cert == 0:
            self.Y_grid = np.tile(self.Y_grid, (len(self.grid_a),1)).T
            O = np.ones((len(self.Y_grid),len(self.grid_a)))
            self.grid_a.shape = len(self.grid_a),1
            self.mesh_a = np.kron(self.grid_a,O)
            self.mesh_Y = np.tile(self.Y_grid, (len(self.grid_a),1))
            self.grid_a.shape = len(self.grid_a),
            self.mesh_ap = np.tile(self.grid_a, (len(self.mesh_Y),1))
            self.C = self.mesh_a*(1+self.r) + self.mesh_Y - self.mesh_ap
            self.Tr_l = self.Tr[:,0]
            self.Tr_l = np.tile(self.Tr_l, (len(self.grid_a),1))
            self.Tr_h = self.Tr[:,1]
            self.Tr_h = np.tile(self.Tr_h, (len(self.grid_a),1))
        elif self.cert == 1:
            self.mesh_ap = np.tile(self.grid_a,(self.N_a,1))
            self.mesh_a = self.mesh_ap.T
            self.C = self.mesh_a*(1+self.r) + 1 - self.mesh_ap
        
        
        
    def markov_Tr(self, N_s, Mu_y = 1, Sig_y = 0.5, gamma=0.7, m=1):
        rho = gamma
        Sig_eps = Sig_y*((1 -rho**2)**(1/2))
        max_y = Mu_y + m*Sig_y
        min_y = Mu_y - m*Sig_y
        Y_grid = np.linspace(min_y, max_y, N_s)
        Mu = Mu_y*(1-rho) 
        w = np.abs(max_y-min_y)/(N_s-1)
        Tr = np.zeros((N_s,N_s))
        if Sig_y == 0:
            Tr = np.eye(N_s)
        else:
            for i in range(N_s):
                for j in range(1,N_s-1):
                    Tr[i,j] = norm.cdf((Y_grid[j] - Mu -rho*Y_grid[i] + w/2)/Sig_eps ) - norm.cdf((Y_grid[j] - Mu -rho*Y_grid[i]-w/2)/Sig_eps )  
                Tr[i,0] = norm.cdf((Y_grid[0] - Mu -rho*Y_grid[i]+w/2)/Sig_eps )
                Tr[i,N_s-1] = 1 - norm.cdf((Y_grid[N_s-1] - Mu -rho*Y_grid[i]-w/2)/Sig_eps)
        return Tr, Y_grid
    
    def update_chi(self, C, V):
        if self.cert == 0:
            Vl = V[:,0]
            Vh = V[:,1]
            E_Vl = self.Tr_l[:,0]*Vl + self.Tr_l[:,1]*Vh
            E_Vh = self.Tr_h[:,0]*Vl + self.Tr_h[:,1]*Vh
        
            E_V = np.vstack((E_Vl, E_Vh))
        elif self.cert == 1:
            E_V = V.copy()
        # V is a matrix
        if self.U2 == 1:
            Chi = U2(C, self.Sig) + self.beta*np.tile(E_V, (len(self.grid_a),1))
        else:
            Chi = U1(C, self.Sig) + self.beta*np.tile(E_V, (len(self.grid_a),1))
        return Chi
    
    def update_V(self, Vold, C, ret = 0):
        Chi = self.update_chi(C, Vold)
        argm_pos = np.argmax(Chi, axis=1)
        V_new = []
        ga = []
        gc = []
        for i, idx in enumerate(list(argm_pos)):
            v = Chi[i,idx]
            g1 = self.mesh_ap[i,idx]
            g2 = C[i,idx] 
            V_new.append(v)
            ga.append(g1)
            gc.append(g2)
        V_new = np.array(V_new)
        ga = np.array(ga)
        gc = np.array(gc)
        if self.cert == 0:
            V_new = np.reshape(V_new, (len(self.grid_a),len(self.Y_grid)))
            ga = np.reshape(ga, (len(self.grid_a),len(self.Y_grid)))
            gc = np.reshape(gc, (len(self.grid_a),len(self.Y_grid)))
        if ret == 1:
            return V_new, ga, gc
        elif ret == 0:
            return V_new

    
    def problem(self):
        V = []
        Ga = []
        Gc = []
        if self.cert == 0:
            V_start = np.zeros((len(self.grid_a), len(self.Y_grid)))
        elif self.cert == 1:
            V_start = np.zeros((len(self.grid_a),))
        V.append(V_start)
        for i in range(self.T):
            V_new, ga, gc = self.update_V(V_start, self.C, ret = 1)
            V.append(V_new)
            Ga.append(ga)
            Gc.append(gc)
            V_start = V_new
            
        return V, Ga, Gc