import numpy as np
from scipy.interpolate import interp2d
from numpy.random import uniform
from numpy import vectorize

########function to lag a vector or a matrix
def lagmat(A, lag, uni = 0):
    if uni==0:
        Old_A = A[:-lag,:]
        for i in range(1,lag+1):
            if i != lag:
                new_A = A[i:-lag+i,:]
                Old_A = np.hstack((Old_A,new_A))
            else:
                new_A = A[i:,:]
                Old_A = np.hstack((Old_A,new_A))
    if uni==1:
        Old_A = A[:-lag]
        for i in range(1,lag+1):
            if i != lag:
                new_A = A[i:-lag+i]
                Old_A = np.vstack((Old_A,new_A))
            else:
                new_A = A[i:]
                Old_A = np.vstack((Old_A,new_A))
        Old_A = Old_A.T
    return np.array(Old_A)


@vectorize
def U(c):
    if c<=0:
        u = -np.inf
    else:
        u = np.log(c)
    return u

class K_S:
    
    def __init__(self, N_k = 100, N_K = 20, N_Z=2, N_eps=2, Ug = 0.04, Ub = 0.1, beta = 0.95, delta = 0.0025, L=1, Guess = None):
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
        self.Piz = np.reshape(x, (4,4)).T
        
        ############## GRIDS #################
        k1 = np.linspace(0.04,5,int(N_k*0.7))
        k2 = np.linspace(5.3,50,N_k - int(N_k*0.7))
        grid_k = np.hstack((k1,k2))
        self.grid_k = grid_k
        
        grid_K = np.linspace(8,20,N_K)
        self.grid_K = grid_K
        ######useful for interpolation######
        self.mesh_K = np.tile(grid_K, (N_k,1))
        self.mesh_k_int = np.tile(grid_k, (N_K,1)).T
        
        
        mesh_k = np.tile(grid_k, (N_k,1)).T
        self.mesh_k = mesh_k
        mesh_kp = mesh_k.T
        self.mesh_kp = mesh_kp
        
        ################### Useful ############
        self.mat = np.array([[0,1],[2,3]])# g row e column, e=0 unemployed, g = 0 good
        ##inverse function mat_1
        self.mat_1 = np.array([[0,0],[0,1],[1,0],[1,1]])
        ############### Choice ################
        self.match_k = np.vectorize(lambda k: np.argmin(self.grid_k - k))
        if Guess is None:
            b0g_old, b1g_old, b0b_old, b1b_old = 0, 1, 0, 1
            self.b0g, self.b1g, self.b0b, self.b1b = b0g_old, b1g_old, b0b_old, b1b_old
            H = np.vectorize(lambda K, zi: np.exp((self.b0g + self.b1g*np.log(K))*zi + (self.b0b + self.b1b*np.log(K))*(1-zi)))
            self.H = H
            Ha = np.vectorize(lambda K, zi: np.argmin(self.grid_K - H(K,zi)))# I want to find the element in the grid that is as close a possible from the perceived law of motion, it gives me aggregate capital in the future
            self.Ha = Ha
            # maybe argmin is makes more sens for what you want to do
            r = np.vectorize(lambda I, e, g: self.alpha*self.z[g]*(self.grid_K[I]/self.L[g])**(self.alpha-1))
            self.r = r
            w = np.vectorize(lambda I, e, g: (1-self.alpha)*self.z[g]*(self.grid_K[I]/self.L[g])**(self.alpha))
            self.w = w
            C = np.vectorize(lambda I, e, g:  (1 - self.delta + r(I, e, g))*self.mesh_k + r(I, e, g)*e - self.mesh_kp)
            self.C = C
        else:
            b0g_old, b1g_old, b0b_old, b1b_old = Guess[0], Guess[1], Guess[2], Guess[3]
            self.b0g, self.b1g, self.b0b, self.b1b = b0g_old, b1g_old, b0b_old, b1b_old
            H = np.vectorize(lambda K, zi: np.exp((self.b0g + self.b1g*np.log(K))*zi + (self.b0b + self.b1b*np.log(K))*(1-zi)))
            self.H = H
            Ha = np.vectorize(lambda K, zi: np.argmin(self.grid_K - H(K,zi)))# I want to find the element in the grid that is as close a possible from the perceived law of motion, it gives me aggregate capital in the future
            self.Ha = Ha
            # maybe argmin is makes more sens for what you want to do
            r = np.vectorize(lambda I, e, g: self.alpha*self.z[g]*(self.grid_K[I]/self.L[g])**(self.alpha-1))
            self.r = r
            w = np.vectorize(lambda I, e, g: (1-self.alpha)*self.z[g]*(self.grid_K[I]/self.L[g])**(self.alpha))
            self.w = w
            C = np.vectorize(lambda I, e, g:  (1 - self.delta + r(I, e, g))*self.mesh_k + r(I, e, g)*e - self.mesh_kp)
            self.C = C
            
        
    def exp_V(self, V, I, g, e): # good and employed are just there to select the row in the transition matrix
        I_new = self.Ha(self.grid_K[I],(1-g)**2)# when good g =0
        #K = self.H(self.grid_K[I],(1-g)**2)
        row = self.mat[g,e]
        T = self.Piz[row,:]
        V0 = np.zeros((len(self.grid_k),))
        for i in range(len(V)):#V is a list of functions
            V0 = np.vstack((V0,V[i](self.grid_k,self.grid_K[I_new])))
            #V0 = np.vstack((V0,V[i](self.grid_k,K)))        
        V0 = V0.T
        V0 = V0[:,1:]
        E_V = T@V0.T
        # you don't need to store the I new the only thing you want is the function
        return E_V
    
    
    def update_V(self, Vold, I, ret = 0):
        g = [0, 1] # I do this because the first elements that you see in the 
        e = [0, 1] # transition matrix a good states of the world so I need
        # to write the good events first in the list
        vec_del = np.zeros((self.N_k,))
        Vo = vec_del.copy()
        gko = vec_del.copy()
        gco = vec_del.copy()
        poso = vec_del.copy()
        
        for ix in g: # g needs to come first otherwise it will not be consistent with the transition matrix
            for j in e:
                E_V = self.exp_V(Vold, I, ix, j)
                E_V_mesh = np.tile(E_V, (self.N_k,1))# you don't have to transpose remember E_V_mesh is function
                #of the future K not the K today hence it needs to be a horizontal vector
                C_gr = self.C(I, j, ix)
                
                X = U(C_gr) + self.beta*E_V_mesh
                argm_pos = np.argmax(X, axis=1)
                V_new = []
                gk = []
                gc = []
                k = 0
                for i in np.nditer(argm_pos): #see if np.nditer works otherwise do a usual loop
                    idx = argm_pos[k]
                    v = X[k,idx]
                    g1 = self.mesh_kp[i,idx]
                    g2 = C_gr[i,idx]
                    V_new.append(v)
                    gk.append(g1)
                    gc.append(g2)
                    k=k+1
                ######stacking the vectors together#######    
                V_new = np.array(V_new)
                Vo = np.vstack((Vo,V_new))
                
                gk = np.array(gk)
                gko = np.vstack((gko,gk))
                
                gc = np.array(gc)
                gco = np.vstack((gco,gc))
                
                poso = np.vstack((poso,argm_pos))
        #######The first row is a column of zeros I don't want it###########    
        Vo = Vo.T
        Vo = Vo[:,1:]
        
        gko = gko.T
        gko = gko[:,1:]
        
        gco = gco.T
        gco = gco[:,1:] 
        
        poso = poso.T
        poso = poso[:,1:]                   
    
        if ret == 1:
            return Vo, gko
        elif ret == 0:
            return Vo
                
    def Biv_V_interp(self, V0, ret=0):
        
        O = np.zeros((len(self.grid_k),))
        V0g_o, V1g_o, V0b_o, V1b_o = O.copy(), O.copy(), O.copy(), O.copy()
        gk_0g_o, gk_1g_o, gk_0b_o, gk_1b_o = O.copy(), O.copy(), O.copy(), O.copy()
        
        for I in range(len(self.grid_K)):
                
            V, gk = self.update_V(V0,I,ret=1)
            
            V0g, V1g, V0b, V1b = V[:,0], V[:,1], V[:,2], V[:,3]
            V0g_o, V1g_o, V0b_o, V1b_o = np.vstack((V0g_o,V0g)), np.vstack((V1g_o,V1g)), np.vstack((V0b_o,V0b)), np.vstack((V1b_o,V1b))
            
            gk_0g, gk_1g, gk_0b, gk_1b = gk[:,0], gk[:,1], gk[:,2], gk[:,3]
            gk_0g_o, gk_1g_o, gk_0b_o, gk_1b_o = np.vstack((gk_0g_o,gk_0g)), np.vstack((gk_1g_o,gk_1g)), np.vstack((gk_0b_o,gk_0b)), np.vstack((gk_1b_o,gk_1b))
        
        
        
        
        V0g_o, V1g_o, V0b_o, V1b_o = V0g_o.T, V1g_o.T, V0b_o.T, V1b_o.T
        V0g_o, V1g_o, V0b_o, V1b_o = V0g_o[:,1:], V1g_o[:,1:], V0b_o[:,1:], V1b_o[:,1:]   
        
        gk_0g_o, gk_1g_o, gk_0b_o, gk_1b_o = gk_0g_o.T, gk_1g_o.T, gk_0b_o.T, gk_1b_o.T
        gk_0g_o, gk_1g_o, gk_0b_o, gk_1b_o = gk_0g_o[:,1:], gk_1g_o[:,1:], gk_0b_o[:,1:], gk_1b_o[:,1:] 
           
        #####not sure 100% that this is right check results if they dont make sens it might come from here#####
        V0g_int = interp2d(self.mesh_k_int, self.mesh_K, V0g_o)
        V1g_int = interp2d(self.mesh_k_int, self.mesh_K, V1g_o)
        V0b_int = interp2d(self.mesh_k_int, self.mesh_K, V0b_o)
        V1b_int = interp2d(self.mesh_k_int, self.mesh_K, V1b_o)
        
        gk0g_int = interp2d(self.mesh_k_int, self.mesh_K, gk_0g_o)
        gk1g_int = interp2d(self.mesh_k_int, self.mesh_K, gk_1g_o)
        gk0b_int = interp2d(self.mesh_k_int, self.mesh_K, gk_0b_o)
        gk1b_int = interp2d(self.mesh_k_int, self.mesh_K, gk_1b_o)
        
        V_int = [V0g_int, V1g_int, V0b_int, V1b_int]
        gk_int = [gk0g_int, gk1g_int, gk0b_int, gk1b_int]
        
        if ret == 1:
            return V_int, gk_int
        elif ret == 0:
            return V_int
                
    
    def Start_VFI(self, Guess = None, Tol = 10**(-2)):
        if Guess is None:
            
            V0g = np.vectorize(lambda k, K: U(self.alpha*self.z[0]*(K/self.L[0])**(1-self.alpha) *k -
                                      self.delta*k)/(1-self.beta))
            
            V1g = np.vectorize(lambda k, K: U(self.alpha*self.z[0]*(K/self.L[0])**(1-self.alpha) *k + 
                                      (1-self.alpha)*self.z[0]*(K/self.L[0])**self.alpha -
                                      self.delta*k)/(1-self.beta))
            
            V0b = np.vectorize(lambda k, K: U(self.alpha*self.z[1]*(K/self.L[1])**(1-self.alpha) *k -
                                      self.delta*k)/(1-self.beta))
            
            V1b = np.vectorize(lambda k, K: U(self.alpha*self.z[1]*(K/self.L[1])**(1-self.alpha) *k + 
                                      (1-self.alpha)*self.z[1]*(K/self.L[1])**self.alpha -
                                      self.delta*k)/(1-self.beta))
            
        
            Vfunc = [V0g, V1g, V0b, V1b]
        else:
            Vfunc = Guess
        
        Vfunc_old = Vfunc 
        err = 1
        j = 0
        while err>Tol:
            Vfunc_new = self.Biv_V_interp(Vfunc_old)
            Err = []
            for i in range(len(Vfunc_new)):
                for k in range(self.N_K):
                    err_i = np.linalg.norm(Vfunc_new[i](self.mesh_k[:,k],self.mesh_K[:,k])-Vfunc_old[i](self.mesh_k[:,k],self.mesh_K[:,k]))/np.linalg.norm(Vfunc_old[i](self.mesh_k[:,k],self.mesh_K[:,k]))
                Err.append(err_i)
            err = max(Err)
                
            Vfunc_old = Vfunc_new
            if j%5==0:
                print('   iteration:', j)
                print('   error:', err) 
            
            j = j+1
        V_int, gk_int = self.Biv_V_interp(Vfunc_old, ret=1)
        return V_int, gk_int
            
            
    
        
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
         
    def simulation_Ag_K(self, gk_int, Z, E, k_ini = 18, ret = 0):
        
        Ti, N = E.shape
            # initial capital 40
        K = [k_ini]
        k = [list(np.ones((N,))*k_ini)]
            
        for i in range(1,Ti):
            k_ap = []
            for j in range(N):
                z = int(Z[i])
                e = int(E[i,j])
                pos = self.mat[z,e]
                ki = gk_int[pos](k[-1][j],K[-1]) #what is the aggregate capital?
                k_ap.append(ki)
            k.append(k_ap)
            Ki = np.mean(np.array(k_ap))
            K.append(Ki)
        K = np.array(K)
        Kg = K[Z==0]
        Kb = K[Z==1]
        if ret == 0:
            return Kg[200:], Kb[200:]
        elif ret == 1:
            return np.array(k)

    def param_update(self, Guess=None, Tol=10**(-2), max_it = 10, eta = 0.1, ret = 0):
           ############### Choice ################
        if Guess is None:
            b0g_old, b1g_old, b0b_old, b1b_old = 0, 1, 0, 1
            b_old = np.array([b0g_old, b1g_old, b0b_old, b1b_old])
            self.b0g, self.b1g, self.b0b, self.b1b = b0g_old, b1g_old, b0b_old, b1b_old
            H = np.vectorize(lambda K, zi: np.exp((self.b0g + self.b1g*np.log(K))*zi + (self.b0b + self.b1b*np.log(K))*(1-zi)))
            self.H = H
            Ha = np.vectorize(lambda K, zi: np.argmin(self.grid_K - H(K,zi)))# I want to find the element in the grid that is as close a possible from the perceived law of motion, it gives me aggregate capital in the future
            self.Ha = Ha
            # maybe argmin is makes more sens for what you want to do
            r = np.vectorize(lambda I, e, g: self.alpha*self.z[g]*(self.grid_K[I]/self.L[g])**(self.alpha-1))
            self.r = r
            w = np.vectorize(lambda I, e, g: (1-self.alpha)*self.z[g]*(self.grid_K[I]/self.L[g])**(self.alpha))
            self.w = w
            C = np.vectorize(lambda I, e, g:  (1 - self.delta + r(I, e, g))*self.mesh_k + r(I, e, g)*e - self.mesh_kp)
            self.C = C
        else:
            b0g_old, b1g_old, b0b_old, b1b_old = Guess[0], Guess[1], Guess[2], Guess[3]
            b_old = np.array([b0g_old, b1g_old, b0b_old, b1b_old])
            self.b0g, self.b1g, self.b0b, self.b1b = b0g_old, b1g_old, b0b_old, b1b_old
            H = np.vectorize(lambda K, zi: np.exp((self.b0g + self.b1g*np.log(K))*zi + (self.b0b + self.b1b*np.log(K))*(1-zi)))
            self.H = H
            Ha = np.vectorize(lambda K, zi: np.argmin(self.grid_K - H(K,zi)))# I want to find the element in the grid that is as close a possible from the perceived law of motion, it gives me aggregate capital in the future
            self.Ha = Ha
            # maybe argmin is makes more sens for what you want to do
            r = np.vectorize(lambda I, e, g: self.alpha*self.z[g]*(self.grid_K[I]/self.L[g])**(self.alpha-1))
            self.r = r
            w = np.vectorize(lambda I, e, g: (1-self.alpha)*self.z[g]*(self.grid_K[I]/self.L[g])**(self.alpha))
            self.w = w
            C = np.vectorize(lambda I, e, g:  (1 - self.delta + r(I, e, g))*self.mesh_k + r(I, e, g)*e - self.mesh_kp)
            self.C = C
        Z, E = self.Simulation_U_Z() # if Z = 0 good if E = 0 employed
        err = 1
        j = 0
        while err>Tol:
            if j == max_it:
                break
            V_int, gk_int = self.Start_VFI()
            Kg_st, Kb_st = self.simulation_Ag_K(gk_int, Z, E)
            lag = 1
            Kg_st_l = lagmat(Kg_st, lag, uni=1)
            Kb_st_l = lagmat(Kb_st, lag, uni=1)
            Xg = np.vstack((np.ones((len(Kg_st_l[:,1]),)),Kg_st_l[:,1])).T
            Bg = np.linalg.inv(Xg.T@Xg)@Xg.T@Kg_st_l[:,0]
            R2g = 1 - np.sum((Kg_st_l[:,0]-Xg@Bg)**2)/np.sum((Kg_st_l[:,0]-np.mean(Kg_st_l[:,0]))**2)
            
            Xb = np.vstack((np.ones((len(Kb_st_l[:,1]),)),Kb_st_l[:,1])).T
            Bb = np.linalg.inv(Xb.T@Xb)@Xb.T@Kb_st_l[:,0]
            R2b = 1 - np.sum((Kb_st_l[:,0]-Xb@Bb)**2)/np.sum((Kb_st_l[:,0]-np.mean(Kb_st_l[:,0]))**2)
            
            b0g_new = (1-eta)*self.b0g + eta*Bg[0]
            b1g_new = (1-eta)*self.b1g + eta*Bg[1]
            b0b_new = (1-eta)*self.b0b + eta*Bb[0]
            b1b_new = (1-eta)*self.b1b + eta*Bb[1]
            b_new = np.array([b0g_new, b1g_new, b0b_new, b1b_new])
            self.b0g, self.b1g, self.b0b, self.b1b = b0g_new, b1g_new, b0b_new, b1b_new
            H = np.vectorize(lambda K, zi: np.exp((self.b0g + self.b1g*np.log(K))*zi + (self.b0b + self.b1b*np.log(K))*(1-zi)))
            self.H = H
            Ha = np.vectorize(lambda K, zi: np.argmin(self.grid_K - H(K,zi)))# I want to find the element in the grid that is as close a possible from the perceived law of motion, it gives me aggregate capital in the future
            self.Ha = Ha
            # maybe argmin is makes more sens for what you want to do
            r = np.vectorize(lambda I, e, g: self.alpha*self.z[g]*(self.grid_K[I]/self.L[g])**(self.alpha-1))
            self.r = r
            w = np.vectorize(lambda I, e, g: (1-self.alpha)*self.z[g]*(self.grid_K[I]/self.L[g])**(self.alpha))
            self.w = w
            C = np.vectorize(lambda I, e, g:  (1 - self.delta + r(I, e, g))*self.mesh_k + r(I, e, g)*e - self.mesh_kp)
            self.C = C
            
            err = np.linalg.norm(b_new-b_old)/np.linalg.norm(b_old)
            if j%1==0:
                print('Parameter Updated:', j)
                print('Parameter Error:', err) 
                print('Parameters:', b_new)
                print('R_squared good', R2g)
                print('R_squared bad', R2b)
            
            j = j+1
            R_squared = [R2g, R2b]
            b_old = b_new
        if ret == 0:
            return b_old, R_squared
        elif ret == 1:
            dist_k = self.simulation_Ag_K(gk_int, Z, E, ret=1)
            return dist_k
            
    def simu_dist(self, gk_int, Ik, state = 'good', T=7, N=1000):
            ###initial distribution unemployement
        I = [list(np.hstack((np.ones(int(N*0.96)),np.zeros(N-int(N*0.96)))))]#remember e=0 means unemployed
        if state == 'good':
            Un = uniform(0,1,(T,N))
            for i in range(1,T):
                g = int(0)
                g_p = int(0)
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
            K = [np.mean(Ik)]
            k = [Ik]
            
            for i in range(1,T):
                k_ap = []
                for j in range(N):
                    z = int(0)
                    e = int(I[i,j])
                    pos = self.mat[z,e]
                    ki = gk_int[pos](k[-1][j],K[-1]) #what is the aggregate capital?
                    k_ap.append(ki)
                k.append(k_ap)
                Ki = np.mean(np.array(k_ap))
                K.append(Ki)
            k = np.array(k)
                
        elif state == 'bad':
            Un = uniform(0,1,(T,N))
            for i in range(1,T):
                g = int(1)
                g_p = int(1)
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
            K = [np.mean(Ik)]
            k = [Ik]
        
            for i in range(1,T):
                k_ap = []
                for j in range(N):
                    z = int(1)
                    e = int(I[i,j])
                    pos = self.mat[z,e]
                    ki = gk_int[pos](k[-1][j],K[-1]) #what is the aggregate capital?
                    k_ap.append(ki)
                k.append(k_ap)
                Ki = np.mean(np.array(k_ap))
                K.append(Ki)
            k = np.array(k)
        return k[-1,:]
    
    def Simu_2ag(self,gk_int_best, gk_int_not, Z, E_ini, k_ini=18, ret=0):
        E = np.tile(E_ini,(1,2))
        Ti, N = E.shape
            # initial capital 40
        K = [k_ini]
        k = [list(np.ones((N,))*k_ini)]
            
        for i in range(1,Ti):
            k_ap = []
            for j in range(N):
                if j>=int(N/2):
                    gk_int = gk_int_best
                elif j<int(N/2):
                    gk_int = gk_int_not
                z = int(Z[i])
                e = int(E[i,j])
                pos = self.mat[z,e]
                ki = gk_int[pos](k[-1][j],K[-1]) #what is the aggregate capital?
                k_ap.append(ki)
            k.append(k_ap)
            Ki = np.mean(np.array(k_ap))
            K.append(Ki)
        K = np.array(K)
        Kg = K[Z==0]
        Kb = K[Z==1]
        if ret == 0:
            return Kg[200:], Kb[200:]
        elif ret == 1:
            return np.array(k), K    
            
               
            
                
        

            
            
        
        