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
    
    def __init__(self, N_k = 100, N_K = 100, N_L= 100, N_Z=2,Gam=3,gam=1.5,N_eps=2, Ug = 0.04, Ub = 0.1, beta = 0.95, delta = 0.0025, L=1, state = 'good', Guess = None):
        self.state = state
        self.Gam, self.gam = Gam, gam 
        self.beta = beta
        self.delta = delta
        self.z = np.array([1.01, 0.99])
        self.alpha = 0.36
        #self.L = np.array([L-Ug, L-Ub])
        self.N_k = N_k
        N_l = N_k
        self.N_l = N_l 
        self.N_K = N_K
        self.N_L = N_L
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
        
        l1 = np.linspace(0,0.95,int(N_l*0.3))
        l2 = np.linspace(0.95+(1/(N_l*0.7)),1,N_l - int(N_l*0.3))
        grid_l = np.hstack((l1,l2))
        self.grid_l = grid_l
        
        ##########################
        grid_K = np.linspace(15,40,N_K)
        self.grid_K = grid_K
        
        # L is the intensive margin not the employement or enemployement
        # for reasonable parameters the intensive margin should be around 0.31 c.f raul
        grid_L = np.linspace(0.3, 1,N_L)
        self.L = grid_L

        ###################
        ###useful functions for matching to the grid###
        H_L = np.vectorize(lambda L: np.argmin(np.abs(self.L-L)))
        self.H_L = H_L
        
        H_K = np.vectorize(lambda K: np.argmin(np.abs(self.grid_K-K)))
        self.H_K = H_K
        
        
        ############################################
        
        mesh_k = np.tile(grid_k, (N_k,1)).T
        self.mesh_k = mesh_k
        mesh_kp = mesh_k.T
        self.mesh_kp = mesh_kp
        
        o = np.ones((N_l,))
        self.mesh_k = np.kron(o,self.mesh_k)
        self.mesh_kp = np.kron(o,self.mesh_kp)
        
        O = np.ones((N_k,N_k))
        self.mesh_n = np.kron(self.grid_l,O)
                    
    def T_endo(self, ga_all):#ga_all need to contain both low and high state
        n,c = ga_all.shape
        Tr = self.Tr
        PHI = []
        PI = []
        o = np.zeros((c,c))
        O = np.zeros((n,n))
        PI = np.zeros((c*n,c*n)) 
        One = np.ones((n,n))
        Mat = []
        for i in range(c):
            for j in range(c):
                mat = o.copy()
                mat[i,j]=1
                Mat.append(mat)
                pi = np.kron(mat,One*Tr[i,j])
                PI = PI+pi 
        PI = PI.T
        
        PHI = []
        for i in range(c):
            pos = ga_all[:,i]
            phi = O.copy()
            for j in range(n):
                phi[j,pos[j]]=1
            PHI.append(phi)
        
        Endo =  np.zeros((c*n,c*n))
        k = 0
        
        for j in range(c): # because it needs to be PHI0.T twice then PHI1.T
            for i in range(c):
                endo = np.kron(Mat[k],PHI[j].T)
                Endo = Endo + endo
                k=k+1
        
        Tendo = PI*Endo
        return Tendo
    
    def Inv_dist(self, ga_all, Tol=10**(-3)):
        Tendo = self.T_endo(ga_all)
        Pold = np.ones(len(Tendo))/len(Tendo)
        err = 1
        while err>Tol:
            Pnew = Tendo@Pold
            err = np.linalg.norm(Pnew-Pold)/np.linalg.norm(Pold)
            Pold = Pnew
            
        return Pold
    
    def U(self,c,n):
        u = U(c,n,self.Gam,self.gam)
        return u
    
    def exp_V(self, V, e):
        if self.state == 'good':
            T = self.P_gg[e,:]
        elif self.state == 'bad':
            T = self.P_bb[e,:]
        E_V = T@V.T
        return E_V
    
    def update_V(self, Vold, I, I_l, ret = 0):
        e = [0, 1]
        V0 = np.zeros((len(self.grid_k),))
        gk0 = np.zeros((len(self.grid_k),))
        gn0 = np.zeros((len(self.grid_k),))
        pos_k0 = np.zeros((len(self.grid_k),))
        pos_n0 = np.zeros((len(self.grid_k),)) 
        
        o = np.ones((self.N_l,))
        for j in e:
#            if j==1:
#                print('here')
            V = []
            gk = []
            gn = []
            pos_k = []
            pos_n = []
            E_V = self.exp_V(Vold, j)
            E_V_mesh = np.tile(E_V, (self.N_k,1))#of the future K not the K today hence it needs to be a horizontal vector
            E_V_mesh = np.kron(o,E_V_mesh)
            C = self.C(I, I_l,j)
            X = self.U(C,self.mesh_n) + self.beta*E_V_mesh
            Pos = np.argmax(X,axis=1)
            for ix in range(self.N_k):
                v = X[ix,Pos[ix]]
                g1 = self.mesh_kp[ix,Pos[ix]]
                g2 = self.mesh_n[ix,Pos[ix]]
#                if ix==41:
#                    print('here')
                g3 = Pos[ix]%self.N_l
                g4 = int(Pos[ix]/self.N_k)# you changed this Pos[ix]//self.N_k
                
                V.append(v)
                gk.append(g1)
                gn.append(g2)
                pos_k.append(g3)
                pos_n.append(g4)
                
            V = np.array(V)
            gk = np.array(gk)
            gn = np.array(gn)
            pos_n = np.array(pos_n)
            pos_k = np.array(pos_k)
            
            
            V0 = np.vstack((V0,V))
            gk0 = np.vstack((gk0,gk))
            gn0 = np.vstack((gn0,gn))
            pos_k0 = np.vstack((pos_k0,pos_k))
            pos_n0 = np.vstack((pos_n0,pos_n))
        
        V0 = V0.T
        gk0 = gk0.T
        gn0 = gn0.T
        pos_k0 = pos_k0.T
        pos_n0 = pos_n0.T
        
        V0 = V0[:,1:]
        gk0 = gk0[:,1:]
        gn0 = gn0[:,1:]
        pos_k0 = pos_k0[:,1:]
        pos_n0 = pos_n0[:,1:]
        
        if ret == 1:
            return V0, gk0, gn0, pos_k0, pos_n0
        elif ret == 0:
            return V0
        elif ret == 'pos':
            return pos_k0, pos_n0, gk0, gn0
      ####CHANGE THE TOLERANCE IT IS TOO HIGH###  
    def Start_VFI(self, I, I_l, Guess = None, Tol = 0.05, max_iter = 80):
        self.g = [0,1]
        if self.state == 'good':
            r = np.vectorize(lambda I, I_l, e: self.alpha*self.z[int(self.g[0])]*(self.grid_K[I]/self.L[I_l])**(self.alpha-1))
            self.r = r
            w = np.vectorize(lambda I, I_l, e: (1-self.alpha)*self.z[int(self.g[0])]*(self.grid_K[I]/self.L[I_l])**(self.alpha))
            self.w = w
            C = np.vectorize(lambda I, I_l, e:  (1 - self.delta + self.r(I, I_l, e))*self.mesh_k + self.w(I, I_l, e)*self.mesh_n*e - self.mesh_kp)
            self.C = C
            self.Tr = self.P_gg.copy()
            
        elif self.state == 'bad':
            r = np.vectorize(lambda I, I_l, e: self.alpha*self.z[int(self.g[1])]*(self.grid_K[I]/self.L[I_l])**(self.alpha-1))
            self.r = r
            w = np.vectorize(lambda I, I_l, e: (1-self.alpha)*self.z[int(self.g[1])]*(self.grid_K[I]/self.L[I_l])**(self.alpha))
            self.w = w
            C = np.vectorize(lambda I, I_l, e:  (1 - self.delta + self.r(I, I_l, e))*self.mesh_k + self.w(I, I_l, e)*self.mesh_n*e - self.mesh_kp)
            self.C = C
            self.Tr = self.P_bb.copy()
        
        if Guess is None:
            if self.state == 'good':
                V0 = np.vectorize(lambda k, K, L: self.U(self.alpha*self.z[0]*(K/L)**(1-self.alpha) *k -
                                                  self.delta*k,0)/(1-self.beta))
                V1 = np.vectorize(lambda k, K, L: self.U(self.alpha*self.z[0]*(K/L)**(1-self.alpha) *k + 
                                                  (1-self.alpha)*self.z[0]*(K/L)**self.alpha - 
                                                  self.delta*k,L)/(1-self.beta))
            elif self.state == 'bad':
                V0 = np.vectorize(lambda k, K, L: self.U(self.alpha*self.z[1]*(K/L)**(1-self.alpha) *k -
                                                 self.delta*k,0)/(1-self.beta))
                V1 = np.vectorize(lambda k, K, L: self.U(self.alpha*self.z[1]*(K/L)**(1-self.alpha) *k +
                                                 (1-self.alpha)*self.z[1]*(K/L)**self.alpha -
                                                 self.delta*k,L)/(1-self.beta))
        else:
            V0, V1 = Guess[0], Guess[1]
        
        V_old = np.vstack((V0(self.grid_k,self.grid_K[I],self.L[I_l]),V1(self.grid_k,self.grid_K[I],self.L[I_l]))).T 
        err = 1
        j = 0
        while err>Tol:
            V_new = self.update_V(V_old,I, I_l)
            err = np.linalg.norm(V_new-V_old)/np.linalg.norm(V_old)                
            V_old = V_new
            if j%5==0:
                print('   iteration:', j)
                print('   error:', err) 
            if j>=max_iter:
                break
            j = j+1
        pos_k, pos_n, gk, gn = self.update_V(V_old, I, I_l, ret='pos')
        return pos_k, pos_n, gk, gn
    #### I want G to adjust faster than g adjust
    def r_w_update(self, maxiter=30, Tol = 0.01, eta = 0.2): #r_min can't be 0 because of the lower bound
        j = 0
        I = int(self.N_K/2)
        I_l = int(self.N_L/2)
        
        K_old = self.grid_K[I]
        L_old = self.L[I_l]
        count =0
        while True:
            if j >maxiter:
                print('############# Warning ! ################')
                print('##### Maximum number of iterations #####')
                print('############# Warning ! ################')
                pos_k, pos_n, gk, gn = self.Start_VFI(I,I_l)
                break

            ########## solve problems good and bad
            pos_k, pos_n, gk, gn = self.Start_VFI(I,I_l)
            
            pos_k = pos_k.astype(int)#to be able to select elements
            dist_k = self.Inv_dist(pos_k)
            pos_n = pos_n.astype(int)#to be able to select elements
            dist_n = self.Inv_dist(pos_n)
            
            nk, ck = pos_k.shape ###
            nn, cn = pos_k.shape ###
            gk_endo = np.reshape(gk.T,(nk*ck,))##after checking reshape I decided to transpose
            gn_endo = np.reshape(gn.T,(nn*cn,))
            
            Ks = dist_k@gk_endo
            Kd = self.grid_K[I]
            Excess_K = Ks - Kd # for market clearing
            
            Ls = dist_n@gn_endo
            Ld = self.L[I_l]
            Excess_L = Ls - Ld # for market clearing
            
            #update capital and labor
            K = eta*Ks + (1-eta)*Kd 
            L = eta*Ls + (1-eta)*Ld # because Ld is our previous guess of aggregate labor
                  ## but we want to select elements from the grid
            I = self.H_K(K)
            I_l = self.H_L(L)###### need to update all the functions with the new I and I_l
            
            Kgrid = self.grid_K[I]
            Lgrid = self.L[I_l]
            
            w = self.w(I,I_l,1) #it doesn't matter 0, 1 it will not affect the function
            r = self.r(I,I_l,1)
            print('RESULT:',j) # Just to check if there is any issue with convergence
            print('Update Capital Grid:',[Kgrid, I])
            print('Excess Capital:',Excess_K)
            print('Update Intensive Grid:',[Lgrid,I_l])
            print('Excess Labor:',Excess_L)
            print('Wage:',w) 
            print('Interest:',r)
            
            if Lgrid == L_old:
                count = count+1
                if count >= 3:
                    break
                if Kgrid == K_old:# I don't want to wait for capital to change utility
                    break 
            
            K_old = Kgrid
            L_old = Lgrid
            j = j+1
            
        dist_lab = [dist_n[:self.N_l],dist_n[self.N_l:]]
        dist_cap = [dist_k[:self.N_k],dist_k[self.N_k:]]

        Structure = {}
        Structure['Optimal Choice Capital'] = pos_k
        Structure['Optimal Choice Labor'] = pos_n
        Structure['Capital Policy'] = gk
        Structure['Intensive Margin Policy'] = gn
        Structure['Aggregate Capital'] = [I, Kgrid]
        Structure['Aggregate Labor'] = [I_l, Lgrid]
        Structure['Distribution Labor'] = dist_lab
        Structure['Distribution Capital'] = dist_cap
        Structure['Interest'] = self.r(I,I_l,1)
        Structure['Wages'] = self.w(I,I_l,1)
        Structure['Gamma'] = self.Gam
        return Structure, Lgrid
    
    
    
        
        