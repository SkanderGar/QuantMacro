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

@vectorize
def Up(C, S, C_, U2):
    if C<=0:
        Upr = np.inf
    else:
        if U2 == 1:
            Upr = C**(-S)
        else:
            Upr = -(C-C_)
    
         
    return Upr

@vectorize
def Up_1(Mat, S, C_, U2):
    if U2 == 1:
        Inv = Mat**(-1/S)
    else:
        Inv = -Mat + C_ 
    return Inv
#@vectorize
#def E_Up_Cp(Mesh_Cp, Tr, beta, r, S, C_, U2):
    
class agent1:
    
    def __init__(self, N_a, T, a0 = 0, N_s=2, rho = 0.06, Sig = 5, C_ = 100, r = 0.04, B=1, U2 = 1):
        self.beta = 1/(1+rho)
        self.Sig = Sig
        self.C_ = C_
        self.r = r
        self.U2 = 1
        self.N_s = N_s
        self.N_a = N_a 
        self.T = T
        self.B = B
        self.a0 = a0
        
    def markov_Tr(self, N_s, Mu_y = 1, Sig_y = 0.5, gamma=0.7, m=1):
        rho = gamma
        Sig_eps = Sig_y*((1 -rho**2)**(1/2))
        max_y = Mu_y + m*Sig_y
        min_y = Mu_y - m*Sig_y
        Y_grid = np.linspace(min_y, max_y, N_s)
        Mu = Mu_y*(1-rho) 
        w = np.abs(max_y-min_y)/(N_s-1)
        Tr = np.zeros((N_s,N_s))
        for i in range(N_s):
            for j in range(1,N_s-1):
                Tr[i,j] = norm.cdf((Y_grid[j] - Mu -rho*Y_grid[i] + w/2)/Sig_eps ) - norm.cdf((Y_grid[j] - Mu -rho*Y_grid[i]-w/2)/Sig_eps )  
            Tr[i,0] = norm.cdf((Y_grid[0] - Mu -rho*Y_grid[i]+w/2)/Sig_eps )
            Tr[i,N_s-1] = 1 - norm.cdf((Y_grid[N_s-1] - Mu -rho*Y_grid[i]-w/2)/Sig_eps)
        return Tr, Y_grid
    
    def exp_Up(self, Cp):
            E_Up_Cp = self.Tr@Up(Cp, self.Sig, self.C_, self.U2)
            return E_Up_Cp
            
    def update_c(self, Cp):
        E_Up_Cp = self.exp_Up(Cp)
        new_C = Up_1(self.beta*(1+self.r)*E_Up_Cp, self.Sig, self.C_, self.U2)
        return new_C
    
    def problem(self):
        Tr, Y_grid = self.markov_Tr(self.N_s)
        self.Tr = Tr
        self.Y_grid = Y_grid
        if self.B == 1:
            A_T = Y_grid[0]*(1/(1+self.r))
        else:
            A_T = 0
        
        ####need endog grid
        
        #max_a = Y_grid[-1]*(1/(1+self.r))
        max_a = self.a0*(1+self.r)**self.T + (1-(1+self.r)**self.T)/(1-(1+self.r))
        min_a = -A_T
        grid_a = np.linspace(min_a, max_a, self.N_a) 
        Mesh_a = np.tile(grid_a, (len(Y_grid),1))
        Mesh_y = np.tile(Y_grid, (len(grid_a),1)).T
         
        ####### last period
        C_store = []
        a_store = []
        C_T = (1+self.r)*Mesh_a + Mesh_y
        a_store.append(Mesh_a)
        C_store.append(C_T)
        a_T = Mesh_a.copy()
        for i in range(self.T):
            if self.B == 1:
                A_T = -Y_grid[0]*(1/(1+self.r))**(i)
            else:
                A_T = 0
            max_a = self.a0*(1+self.r)**(self.T-(i+1)) + (1-(1+self.r)**(self.T-(i+1)))/(1-(1+self.r))
            C_T_1 = self.update_c(C_T)
            a_T_1 = (C_T_1 + a_T-Mesh_y)/(1+self.r)
            ax1 = (a_T_1 <= -A_T) ###endogenous grid method
            ax2 = (a_T_1 >= max_a) #want to make sure in period 0 people consume what they have
            ### getting consumption by inverting the U can give me savings that don't
            ## satisfy my constraint hence ax is the set of saving that are under 
            ##my lower bound
            a_T_1[ax1] = -A_T
            a_T_1[ax2] = max_a
            # updated consumption if binding
            C_T_1 = a_T_1*(1+self.r) - a_T + Mesh_y
            
            C_store.append(C_T_1)
            a_store.append(a_T_1)
            
            C_T = C_T_1
            a_T = a_T_1
            
        return C_store, a_store
            
        
        
        
        
        
        
        
        
        