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
    
    def __init__(self, N_k = 100, N_K = 100, N_L= 100, N_Z=2,Gam=2.5,gam=2.5,N_eps=2, Ug = 0.04, Ub = 0.1, beta = 0.99, delta = 0.0025, L=1, state = 'good', Guess = None):
        self.state = state
        self.Gam, self.gam = Gam, gam 
        self.beta = beta
        self.delta = delta
        self.z = np.array([1.01, 0.99])
        self.alpha = 0.36
        #self.L = np.array([L-Ug, L-Ub])
        self.N_k = N_k
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
        ##########################
        grid_K = np.linspace(3,20,N_K)
        self.grid_K = grid_K
        
        # L is the intensive margin not the employement or enemployement
        # for reasonable parameters the intensive margin should be around 0.31 c.f raul
        grid_L = np.linspace(0.3, 1,N_L)
        self.L = grid_L
        
        ######useful for interpolation######
        self.mesh_K = np.tile(grid_K, (N_k,1))
        self.mesh_k_int = np.tile(grid_k, (N_K,1)).T
        
        ###useful functions###
        ###I ended up not using them but they can be useful for next time
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
        
        self.match_k = np.vectorize(lambda k: np.argmin(self.grid_k - k))
        self.mesh_K = np.vectorize(lambda K: np.argmin(self.grid_K - K))
            
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
    
    def update_V(self, Vold, I, I_l, ret = 0, step = 0):
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
                #C_init = self.C_init(I, I_l,j)
                V = []
                gc = []
                gk = []
                Pos = []
                pos = 0
                for ix in range(len(self.grid_k)):
                    x_old = -np.inf
                    c_old = 0.5
                    ####using monotonicity
                    for jx in range(pos,len(self.grid_k)):
                        BC = np.vectorize(lambda c: self.BC(c, self.mesh_kp[ix,jx], self.mesh_k[ix,jx], I, I_l, j))
                        #C_init[ix,jx] this is what you used to put in fsolve as initial value
                        if j==1:
                            c_new = fsolve(BC, c_old)
                        elif j==0:
                            c_new = (1+self.r(I, I_l, j))*self.mesh_k[ix,jx]-self.mesh_kp[ix,jx]
                        n_new = self.N(c_new,I, I_l,j)
                        x_new = self.U(c_new, n_new) + self.beta*E_V_mesh[ix,jx]
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
            
                gn = self.N(gc,I,I_l,j)
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
                V = self.U(self.gc[:,j],self.gn[:,j]) + self.beta*E_V
                V0 = np.vstack((V0,V))
            V0 = V0.T                         
            V0 = V0[:,1:]
        
        
        if ret == 1:
            return V0, gk0, gc0, gn0, Pos0
        elif ret == 0:
            return V0
        elif ret == 'pos':
            return Pos0, gk0, gn0
      ####CHANGE THE TOLERANCE IT IS TOO HIGH###  
    def Start_VFI(self, I, I_l, Guess = None, Tol = 0.05, step_mod = 1, state = 'good', max_iter = 80):
        self.g = [0,1]
        self.state = state
        if self.state == 'good':
            r = np.vectorize(lambda I, I_l, e: self.alpha*self.z[int(self.g[0])]*(self.grid_K[I]/self.L[I_l])**(self.alpha-1))
            self.r = r
            w = np.vectorize(lambda I, I_l, e: (1-self.alpha)*self.z[int(self.g[0])]*(self.grid_K[I]/self.L[I_l])**(self.alpha))
            self.w = w
            C_init = np.vectorize(lambda I, I_l, e:  (1 - self.delta + self.r(I, I_l, e))*self.mesh_k + self.w(I, I_l, e)*e - self.mesh_kp)
            self.C_init = C_init
            N = np.vectorize(lambda c, I, I_l, e: ((self.w(I, I_l, e)*e)/(self.Gam*c))**(1/self.gam))
            self.N = N
            BC = np.vectorize(lambda c, kp, k, I, I_l, e: c + kp -(1+self.r(I, I_l, e)-self.delta)*k - self.w(I, I_l, e)*e*self.N(c,I, I_l, e) )
            self.BC = BC
            self.Tr = self.P_gg.copy()
            
        elif self.state == 'bad':
            r = np.vectorize(lambda I, I_l, e: self.alpha*self.z[int(self.g[1])]*(self.grid_K[I]/self.L[I_l])**(self.alpha-1))
            self.r = r
            w = np.vectorize(lambda I, I_l, e: (1-self.alpha)*self.z[int(self.g[1])]*(self.grid_K[I]/self.L[I_l])**(self.alpha))
            self.w = w
            C_init = np.vectorize(lambda I, I_l, e:  (1 - self.delta + self.r(I, I_l, e))*self.mesh_k + self.w(I, I_l, e)*e - self.mesh_kp)
            self.C_init = C_init
            N = np.vectorize(lambda c, I, I_l, e: ((self.w(I, I_l, e)*e)/(self.Gam*c))**(1/self.gam))
            self.N = N
            BC = np.vectorize(lambda c, kp, k, I, I_l, e: c + kp -(1+self.r(I, I_l, e)-self.delta)*k - self.w(I, I_l, e)*e*self.N(c,I, I_l,e) )
            self.BC = BC
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
            if j%step_mod==0:
                V_new = self.update_V(V_old,I, I_l,step=0)
            else:
                V_new = self.update_V(V_old,I, I_l,step=1)
            err = np.linalg.norm(V_new-V_old)/np.linalg.norm(V_old)                
            V_old = V_new
            if j%5==0:
                print('   iteration:', j)
                print('   error:', err) 
            if j>=max_iter:
                break
            j = j+1
        Pos, gk, gn = self.update_V(V_old, I, I_l, ret='pos')
        return Pos, gk, gn
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
                Pos, gk, gn = self.Start_VFI(I,I_l)
                break

            ########## solve problems good and bad
            Pos, gk, gn = self.Start_VFI(I,I_l)
            
            Pos = Pos.astype(int)#to be able to select elements
            dist = self.Inv_dist(Pos)
            n, c = Pos.shape ###
            gk_endo = np.reshape(gk.T,(n*c,))##after checking reshape I decided to transpose
            gn_endo = np.reshape(gn.T,(n*c,))
            
            Ks = dist@gk_endo
            Kd = self.grid_K[I]
            Excess_K = Ks - Kd # for market clearing
            
            Ls = dist@gn_endo
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
            
        dist_u = dist[:self.N_k]
        dist_e = dist[self.N_k:]

        Structure = {}
        Structure['Optimal Choice'] = Pos
        Structure['Capital Policy Good'] = gk
        Structure['Intensive Margin Policy Good'] = gn
        Structure['Aggregate Capital Good'] = [I, Kgrid]
        Structure['Aggregate Labor Good'] = [I_l, Lgrid]
        Structure['Distribution Unemployed Good'] = dist_u
        Structure['Distribution Employed Good'] = dist_e
        Structure['Interest Good'] = self.r(I,I_l,1)
        Structure['Wages Good'] = self.w(I,I_l,1)
        Structure['Gamma'] = self.Gam
        return Structure, Lgrid
    
    
    
        
        