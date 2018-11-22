import numpy as np
from scipy.optimize import fsolve
#from numpy.random import uniform
from numpy import vectorize

@vectorize
def U(c,n, Gam,gam):
    if c<=0 or n<0 or n>1:
        u = -np.inf
    else:
        u = np.log(c) - (Gam*n**(1+gam))/(1+gam) 
    return u

class K_S:
    
    def __init__(self, N_k = 100, N_K = 20, N_Z=2, Gam = 1, gam = 1,N_eps=2, Ug = 0.04, Ub = 0.1, beta = 0.95, delta = 0.0025, L=1, state = 'good', Guess = None):
        self.state = state
        self.Gam = Gam
        self.gam = gam
        self.beta = beta
        self.delta = delta
        self.z = np.array([1.01, 0.99])
        self.alpha = 0.36
        self.L = np.array([L-Ug, L-Ub])
        self.N_k = N_k
        self.N_K = N_K
        self.N_Z = N_Z
        self.N_eps = N_eps
        self.Ub = Ub
        self.Ug = Ug
        
        ################## Transition ####################
        T_ag = np.array([[7/8,1/8],[1/8,7/8]])
        self.T_ag = T_ag
        N_comb = N_Z*N_eps
        A = np.zeros((N_comb**2, N_comb**2))
        A[0,0], A[0,4], A[1,1], A[1,5], A[2,10], A[2,14] = 1, 1, 1, 1, 1, 1
        A[3,11], A[3,15], A[4,8], A[4,12], A[5,9], A[5,13] = 1, 1, 1, 1, 1, 1
        A[6,2], A[6,6], A[7,3], A[7,7] = 1, 1, 1, 1
        A[8,0], A[9,10] = 1, 1
        A[10,8], A[10,10], A[11,0], A[11,2] = 5.6, -1, -1, 28/3
        A[12,0], A[12,1], A[12,2], A[12,3] = 0.02, 0.48, 0.05, 0.45
        A[13,1] = 1 
        A[14,8], A[14,9], A[14,10], A[14,11] = 0.02, 0.48, 0.05, 0.45
        A[15,11] = 1
        
        b = np.array([7/8, 7/8, 7/8, 7/8, 1/8, 1/8, 1/8, 1/8, 7/24, 21/40, 0, 0, 
                      0.02, 0.005, 0.05, 0.02])
        x = np.linalg.inv(A)@b
        Piz = np.reshape(x, (4,4)).T
        self.Piz = Piz
        
        self.P_gg = Piz[:2,:2]
        self.P_gg = self.P_gg/np.tile(np.sum(self.P_gg,axis=1),(2,1)).T
        self.P_bb = Piz[2:,2:]
        self.P_bb = self.P_bb/np.tile(np.sum(self.P_bb,axis=1),(2,1)).T
        ############## GRIDS #################
        k1 = np.linspace(0.04,5,int(N_k*0.7))
        k2 = np.linspace(5.3,50,N_k - int(N_k*0.7))
        grid_k = np.hstack((k1,k2))
        #grid_k = np.linspace(0.04,50, N_k)
        self.grid_k = grid_k
        
        grid_K = np.linspace(3,20,N_K)
        self.grid_K = grid_K
        ######useful for interpolation######
        self.mesh_K = np.tile(grid_K, (N_k,1))
        self.mesh_k_int = np.tile(grid_k, (N_K,1)).T
        
        #useful functions
        def flb(ix):
            if ix-5>=0:
                lb = ix-5
            elif ix-5<0:
                lb = 0
            return int(lb)
        self.f_lb = lambda ix: flb(ix)
        
        def fub(ix,u):
            if ix+5<=u:
                ub = ix+5
            elif ix+5>u:
                ub = u
            return int(ub)
        self.f_ub = lambda ix,u: fub(ix,u)
        
        
        mesh_k = np.tile(grid_k, (N_k,1)).T
        self.mesh_k = mesh_k
        mesh_kp = mesh_k.T
        self.mesh_kp = mesh_kp
        
        self.match_k = np.vectorize(lambda k: np.argmin(self.grid_k - k))
        self.mesh_K = np.vectorize(lambda K: np.argmin(self.grid_K - K))
        self.g = [0,1]
        if self.state == 'good':
            r = np.vectorize(lambda I, e: self.alpha*self.z[int(self.g[0])]*(self.grid_K[I]/self.L[int(self.g[0])])**(self.alpha-1))
            self.r = r
            w = np.vectorize(lambda I, e: (1-self.alpha)*self.z[int(self.g[0])]*(self.grid_K[I]/self.L[int(self.g[0])])**(self.alpha))
            self.w = w
            C_init = np.vectorize(lambda I, e:  (1 - self.delta + self.r(I, e))*self.mesh_k + self.w(I, e)*e - self.mesh_kp)
            self.C_init = C_init
            N = np.vectorize(lambda c, I, e: ((self.w(I,e)*e)/(self.Gam*c))**(1/self.gam))
            self.N = N
            BC = np.vectorize(lambda c, kp, k, I, e: c + kp -(1+self.r(I, e))*k - self.w(I, e)*e*self.N(c,I,e) )
            self.BC = BC
            
        elif self.state == 'bad':
            r = np.vectorize(lambda I, e: self.alpha*self.z[int(self.g[1])]*(self.grid_K[I]/self.L[int(self.g[1])])**(self.alpha-1))
            self.r = r
            w = np.vectorize(lambda I, e: (1-self.alpha)*self.z[int(self.g[1])]*(self.grid_K[I]/self.L[int(self.g[1])])**(self.alpha))
            self.w = w
            C_init = np.vectorize(lambda I, e:  (1 - self.delta + self.r(I, e))*self.mesh_k + self.w(I, e)*e - self.mesh_kp)
            self.C_init = C_init
            N = np.vectorize(lambda c, I, e: ((self.w(I,e)*e)/(self.Gam*c))**(1/self.gam))
            self.N = N
            BC = np.vectorize(lambda c, kp, k, I, e: c + kp -(1+self.r(I, e))*k - self.w(I, e)*e*self.N(c,I,e) )
            self.BC = BC
            
    def exp_V(self, V, e):
        if self.state == 'good':
            T = self.P_gg[e,:]
        elif self.state == 'bad':
            T = self.P_bb[e,:]
        E_V = T@V.T
        return E_V
    
    def update_V(self, Vold, I, ret = 0, step = 0):
        e = [0, 1]
        V0 = np.zeros((len(self.grid_k),))
        gk0 = np.zeros((len(self.grid_k),))
        gc0 = np.zeros((len(self.grid_k),))
        gn0 = np.zeros((len(self.grid_k),))
        Pos0 = np.zeros((len(self.grid_k),)) 
        
        if step ==0:
            for j in e:
                E_V = self.exp_V(Vold, j)
                E_V_mesh = np.tile(E_V, (self.N_k,1))#of the future K not the K today hence it needs to be a horizontal vector
                C_init = self.C_init(I,j)
                V = []
                gc = []
                gk = []
                Pos = []
                pos = 0
                for ix in range(len(self.grid_k)):
                    x_old = -np.inf
                    c_old = 0
                    ####using monotonicity
                    for jx in range(pos,len(self.grid_k)):
                        BC = np.vectorize(lambda c: self.BC(c, self.mesh_kp[ix,jx], self.mesh_k[ix,jx], I, j))
                        c_new = fsolve(BC, C_init[ix,jx])
                        n_new = self.N(c_new,I,j)
                        x_new = U(c_new, n_new, self.Gam, self.gam) + self.beta*E_V_mesh[ix,jx]
                        if x_new<x_old:
                            pos = int(jx-1)
                            x = x_old
                            c = c_old
                            break
                        x_old = x_new
                        c_old = c_new
    
                    V.append(x)
                    gc.append(c)
                    gk.append(self.mesh_kp[ix,pos])
                    Pos.append(pos)
            
                V = np.array(V)
                V.shape = len(self.grid_k,)
            
                gc = np.array(gc)
                gc.shape = len(self.grid_k,)
                
                gk = np.array(gk)
                gk.shape = len(self.grid_k,)
            
                gn = self.N(gc,I,j)
                gn.shape = len(self.grid_k,)
            
                Pos = np.array(Pos)
                Pos.shape = len(self.grid_k,)            
                
                V0 = np.vstack((V0,V))
            
                gk0 = np.vstack((gk0,gk))
             
                gc0 = np.vstack((gc0,gc))
            
                gn0 = np.vstack((gn0,gn))
                            
                Pos0 = np.vstack((Pos0,Pos))
        
            V0 = V0.T
            gk0 = gk0.T
            gc0 = gc0.T
            gn0 = gn0.T
            Pos0 = Pos0.T
        
            V0 = V0[:,1:]
            gk0 = gk0[:,1:]
            gc0 = gc0[:,1:]
            gn0 = gn0[:,1:]
            Pos0 = Pos0[:,1:]
            
            self.gc = gc0
            self.gk = gk0
            self.gn = gn0
        elif step ==1:
            for j in e:
                E_V = self.exp_V(Vold, j)
                V = U(self.gc[:,j],self.gn[:,j], self.Gam,self.gam) + self.beta*E_V
                V0 = np.vstack((V0,V))
            V0 = V0.T                         
            V0 = V0[:,1:]
        
        
        if ret == 1:
            return V0, gk0, gc0, gn0, Pos0
        elif ret == 0:
            return V0
        
    def Start_VFI(self, I = int(5), Guess = None, Tol = 10**(-2), step_mod = 1):
        
        if Guess is None:
            if self.state == 'good':
                V0 = np.vectorize(lambda k, K: U(self.alpha*self.z[0]*(K/self.L[0])**(1-self.alpha) *k -
                                                  self.delta*k,0, self.Gam,self.gam)/(1-self.beta))
                V1 = np.vectorize(lambda k, K: U(self.alpha*self.z[0]*(K/self.L[0])**(1-self.alpha) *k + 
                                                  (1-self.alpha)*self.z[0]*(K/self.L[0])**self.alpha - 
                                                  self.delta*k,self.L[0], self.Gam,self.gam)/(1-self.beta))
            elif self.state == 'bad':
                V0 = np.vectorize(lambda k, K: U(self.alpha*self.z[1]*(K/self.L[1])**(1-self.alpha) *k -
                                                 self.delta*k,0, self.Gam,self.gam)/(1-self.beta))
                V1 = np.vectorize(lambda k, K: U(self.alpha*self.z[1]*(K/self.L[1])**(1-self.alpha) *k +
                                                 (1-self.alpha)*self.z[1]*(K/self.L[1])**self.alpha -
                                                 self.delta*k,self.L[1], self.Gam,self.gam)/(1-self.beta))
        else:
            V0, V1 = Guess[0], Guess[1]
        
        V_old = np.vstack((V0(self.grid_k,self.grid_K[I]),V1(self.grid_k,self.grid_K[I]))).T 
        err = 1
        j = 0
        while err>Tol:
            if j%step_mod==0:
                V_new = self.update_V(V_old,I,step=0)
            else:
                V_new = self.update_V(V_old,I,step=1)
            err = np.linalg.norm(V_new-V_old)/np.linalg.norm(V_old)                
            V_old = V_new
            if j%5==0:
                print('   iteration:', j)
                print('   error:', err) 
            j = j+1
        V, gk, gc, gn, Pos = self.update_V(V_old, I, ret=1)
        return V, gk, gc, gn, Pos
        
        