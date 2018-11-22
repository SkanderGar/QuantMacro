import numpy as np
from scipy.optimize import fsolve
from numpy.random import uniform
from numpy import vectorize

#Gamma: 0.5625
#gamma: 2.0
#Aggragate Labor Good 0.9505050505050505
#Aggragate Labor Good 0.908080808080808

@vectorize
def U(c,n, Gam,gam):
    if c<=0 or n<0 or n>1:
        u = -np.inf
    else:
        u = np.log(c) - (Gam*n**(1+gam))/(1+gam) 
    return u
class K_S:
    
    def __init__(self, N_k = 100, N_K = 30, N_L = 30, N_Z=2, N_eps=2, Ug = 0.04, Ub = 0.1, Gam =0.5625, gam=2.5, beta = 0.95, delta = 0.0025, L=1, Guess = None):
        self.beta = beta
        self.delta = delta
        self.Gam = Gam
        self.gam = gam
        self.z = np.array([1.01, 0.99])
        self.alpha = 0.36
        self.L = np.array([0.95, 0.91])
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
        self.Piz = np.reshape(x, (4,4)).T
        
        ############## GRIDS #################
        k1 = np.linspace(0.04,5,int(N_k*0.7))
        k2 = np.linspace(5.3,50,N_k - int(N_k*0.7))
        grid_k = np.hstack((k1,k2))
        self.grid_k = grid_k
        
        l1 = np.linspace(0,0.85,int(N_l*0.3))
        l2 = np.linspace(0.85+(1/N_l),1,N_l - int(N_l*0.3))
        grid_l = np.hstack((l1,l2))
        self.grid_l = grid_l
        
        grid_K = np.linspace(15,40,N_K)
        self.grid_K = grid_K
        
        
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
        ################### Useful ############
        self.mat = np.array([[0,1],[2,3]])
        self.mat_1 = np.array([[0,0],[0,1],[1,0],[1,1]])
        ############### Choice ################
        ####you can use this to find pos it is more secure if you want to change
        #the code
        self.match_k = np.vectorize(lambda k: np.argmin(np.abs(self.grid_k - k)))
        self.match_n = np.vectorize(lambda n: np.argmin(np.abs(self.grid_l - n)))
        
        b0g_old, b1g_old, b0b_old, b1b_old = 0.123, 0.951, 0.114, 0.953
        self.b0g, self.b1g, self.b0b, self.b1b = b0g_old, b1g_old, b0b_old, b1b_old
        H = np.vectorize(lambda K, zi: np.exp((self.b0g + self.b1g*np.log(K))*zi + (self.b0b + self.b1b*np.log(K))*(1-zi)))
        self.H = H
        Ha = np.vectorize(lambda K, zi: np.argmin(self.grid_K - H(K,zi)))# I want to find the element in the grid that is as close a possible from the perceived law of motion, it gives me aggregate capital in the future
        self.Ha = Ha
        r = np.vectorize(lambda I, e, g: self.alpha*self.z[g]*(self.grid_K[I]/self.L[g])**(self.alpha-1))
        self.r = r
        w = np.vectorize(lambda I, e, g: (1-self.alpha)*self.z[g]*(self.grid_K[I]/self.L[g])**(self.alpha))
        self.w = w
        C = np.vectorize(lambda I, e, g:  (1 - self.delta + r(I, e, g))*self.mesh_k + w(I, e, g)*self.mesh_n*e - self.mesh_kp)
        self.C = C
    
    def U(self,c,n):
        u = U(c,n,self.Gam,self.gam)
        return u        
        
    def exp_V(self, V, g, e): 
        row = self.mat[g,e]
        T = self.Piz[row,:]
        E_V = T@V.T
        return E_V
    
    def update_V(self, I, V_Mat_old, ret = 0):
        g = [0, 1]
        e = [0, 1]
        V0 = np.zeros((len(self.grid_k),))
        gk0 = np.zeros((len(self.grid_k),))
        gn0 = np.zeros((len(self.grid_k),))
        pos_k0 = np.zeros((len(self.grid_k),))
        pos_n0 = np.zeros((len(self.grid_k),)) 
        
        o = np.ones((self.N_l,))
        for i in g:
            I_tp1 = self.Ha(self.grid_K[I],(1-i)**2)
            for j in e:

                V = []
                gk = []
                gn = []
                pos_k = []
                pos_n = []
                Vold_p = V_Mat_old[I_tp1]
                E_V = self.exp_V(Vold_p, i, j)
                E_V_mesh = np.tile(E_V, (self.N_k,1))#of the future K not the K today hence it needs to be a horizontal vector
                E_V_mesh = np.kron(o,E_V_mesh)
                C = self.C(I, j, i)
                X = self.U(C,self.mesh_n) + self.beta*E_V_mesh
                Pos = np.argmax(X,axis=1)
                for ix in range(self.N_k):
                    v = X[ix,Pos[ix]]
                    g1 = self.mesh_kp[ix,Pos[ix]]
                    g2 = self.mesh_n[ix,Pos[ix]]
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
    
    def K_up_V(self, V_Mat_old, ret=0): 
        if ret == 1:
            V_Mat, gk_Mat, gn_Mat, pos_k_Mat, pos_n_Mat = [], [], [], [], []
            V, gk, gn, pos_k, pos_n = self.update_V(0,V_Mat_old,ret = 1)
            V_Mat.append(V)
            gk_Mat.append(gk)
            gn_Mat.append(gn)
            pos_k_Mat.append(pos_k)
            pos_n_Mat.append(pos_n)
            for I in range(1,self.N_K):
                V, gk, gn, pos_k, pos_n = self.update_V(I,V_Mat_old,ret = 1)
                V_Mat.append(V)
                gk_Mat.append(gk)
                gn_Mat.append(gn)
                pos_k_Mat.append(pos_k)
                pos_n_Mat.append(pos_n)
            return V_Mat, gk_Mat, gn_Mat, pos_k_Mat, pos_n_Mat
        elif ret == 0:
            V_Mat = []
            V = self.update_V(0,V_Mat_old,ret = 0)
            V_Mat.append(V)
            for I in range(1,self.N_K):
                V = self.update_V(I,V_Mat_old,ret = 0)
                V_Mat.append(V)
            return V_Mat
            
    
    def Start_VFI(self, Tol = 10**(-3), max_iter = 80):
        Vg0 = np.vectorize(lambda k, K: self.U(self.alpha*self.z[0]*(K/self.L[0])**(1-self.alpha) *k -
                                               self.delta*k,0)/(1-self.beta))
        Vg1 = np.vectorize(lambda k, K: self.U(self.alpha*self.z[0]*(K/self.L[0])**(1-self.alpha) *k + 
                                               (1-self.alpha)*self.z[0]*(K/self.L[0])**self.alpha - 
                                               self.delta*k,self.L[0])/(1-self.beta))
        Vb0 = np.vectorize(lambda k, K: self.U(self.alpha*self.z[1]*(K/self.L[1])**(1-self.alpha) *k -
                                               self.delta*k,0)/(1-self.beta))
        Vb1 = np.vectorize(lambda k, K: self.U(self.alpha*self.z[1]*(K/self.L[1])**(1-self.alpha) *k +
                                               (1-self.alpha)*self.z[1]*(K/self.L[1])**self.alpha -
                                               self.delta*k,self.L[1])/(1-self.beta))
        V_Mat_old = []
        for ix in range(self.N_K):
            V_old = np.vstack((Vg0(self.grid_k,self.grid_K[ix]),Vg1(self.grid_k,self.grid_K[ix]),Vb0(self.grid_k,self.grid_K[ix]),Vb1(self.grid_k,self.grid_K[ix]))).T 
            V_Mat_old.append(V_old)
            
        err = 1
        j = 0
        while err>Tol:
            V_Mat_new = self.K_up_V(V_Mat_old, ret=0)
            Err = []
            for i in range(self.N_K):
                err = np.linalg.norm(V_Mat_new[i]-V_Mat_old[i])/np.linalg.norm(V_Mat_old[i])
                Err.append(err)
            err = max(Err)
                            
            V_Mat_old = V_Mat_new
            if j%1==0:
                print('   iteration:', j)
                print('   error:', err) 
            if j>=max_iter:
                break
            j = j+1
        V_Mat, gk_Mat, gn_Mat, pos_k_Mat, pos_n_Mat = self.K_up_V(V_Mat_old,ret=1)
        return V_Mat, gk_Mat, gn_Mat, pos_k_Mat, pos_n_Mat
    
    def Simulation_U_Z(self, Ti = 2200, N = 1000):
        ############ simulate good and bad state############
        Tr = self.T_ag
        Tr_cum = np.cumsum(Tr, axis = 1)
        
        U = uniform(0,1,Ti)
        g = [0, 1]
        start = g[0] # first element is the good element
        st_pos = start
        Pos = []
        for i in range(Ti): 
            prob = Tr[st_pos,:]
            prob_cum = Tr_cum[st_pos,:]
            ### next store the positions in the transition mat which gives you g and e
            for j in range(len(prob)):
                if j == 0:
                    if U[i] <= prob_cum[j]:
                        st_pos = j
                        break
                if U[i] <= prob_cum[j] and prob_cum[j-1] < U[i]:
                    st_pos = j
                    break
            Pos.append(st_pos)
        Pos = np.array(Pos)# pos store the g=0 means good event
        
        ######### Simulate unemployement states #############
        
        ###initial distribution unemployement
        I = [list(np.hstack((np.ones(int(N*0.96)),np.zeros(N-int(N*0.96)))))]#remember e=0 means unemployed
        #zeros mean employed because it is the first position in the matrix
#        list_idx = np.arange(N)
        Un = uniform(0,1,(Ti,N))
        for i in range(1,Ti):
            g = int(Pos[i-1])
            g_p = int(Pos[i])# if g = 0 then good
            I_a = np.zeros((N,))
            for j in range(N):             
                today = self.mat[g,int(I[-1][j])]
                tomorrow = self.mat[g_p,0] # g=0,e=0 is good unemployed
                p = self.Piz[today,tomorrow]/self.T_ag[g,g_p]#Pr(eps'=0|z',z,eps)
                if Un[i,j]<=p:
                    I_a[j] = 0
                else:
                    I_a[j] = 1
            I.append(list(I_a))
        I = np.array(I)     
        return Pos, I
         
    def simulation_Ag_K(self, gk_Mat, gn_Mat, Z, E, k_ini = 18, ret = 0):
        
        Ti, N = E.shape
            # initial capital 40
        K = [k_ini]
        k = [list(np.ones((N,))*k_ini)]
        n = [list(E[0,:])]
        r_vec = []
        w_vec = []
        g = Z[0]
        I = self.H_K(K[-1])
        I_p = self.Ha(self.grid_K[I],(1-g)**2)
        r_p = self.r(I_p,1,g)
        w = self.w(I,1,g)
        r_vec.append(r_p)
        w_vec.append(w)
        for i in range(1,Ti):
            k_ap = []
            n_t = []
            for j in range(N):
                z = int(Z[i])
                e = int(E[i,j])
                pos = self.mat[z,e]
                I_k = self.match_k(k[-1][j])
                ki = gk_Mat[I][I_k,pos] #what is the aggregate capital?
                ni = gn_Mat[I][I_k,pos]
                k_ap.append(ki)
                n_t.append(ni)
            k.append(k_ap)
            n.append(n_t)
            Ki = np.mean(np.array(k_ap))
            K.append(Ki)
            g = Z[i]
            I = self.H_K(K[-1])
            I_p = self.Ha(self.grid_K[I],g)
            r_p = self.r(I_p,1,g)
            w = self.w(I,1,g)
            r_vec.append(r_p)#agents forecast interest rate for tomorrow not today cause they know it with certainty
            w_vec.append(w)
        K = np.array(K)
        Kg = K[Z==0]
        Kb = K[Z==1]
        if ret == 0:
            return Kg[200:], Kb[200:]
        elif ret == 1:
            return np.array(k), np.array(n), np.array(r_vec), np.array(w_vec)
    
    
    