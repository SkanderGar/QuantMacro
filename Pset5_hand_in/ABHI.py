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
    
    def __init__(self, N_a, Mu_y = 1, sig_y=0.5, gamma_y=0.7, T=45, N_s=2, order = 7, delta = 0.015, theta=0.68, rho = 0.06, Sig = 5, C_ = 1 , U2 = 1, B=0):
        self.theta = theta
        self.delta = delta
        self.order = order
        self.T = T
        self.gamma_y = gamma_y
        self.sig_y = sig_y
        self.beta = 1/(1+rho)
        self.Sig = Sig
        self.C_ = C_
        self.U2 = 1
        self.N_s = N_s
        self.N_a = N_a
        self.B = B
        self.Tr, self.Y_grid_s = self.markov_Tr(self.N_s,  Mu_y = Mu_y, Sig_y = self.sig_y, gamma = self.gamma_y)
        self.Tr_l = self.Tr[:,0]
        self.Tr_l = np.tile(self.Tr_l, (N_a,1))
        self.Tr_h = self.Tr[:,1]
        self.Tr_h = np.tile(self.Tr_h, (N_a,1))
        
        func = []
        Phi1 = np.vectorize(lambda x: 1)
        Phi2 = np.vectorize(lambda x: x)
        func.append(Phi1)
        func.append(Phi2)
        if self.order>= 2:
            for i in range(2,self.order):
                f = np.vectorize(lambda x, n=i: 2*func[n-1](x)*x - func[n-2](x))
                func.append(f)
        self.func = func

        
        
        
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
        
    def select_node(self, num, grid):
        n = len(grid)
        element = (n-1)/(num-1)
        values = []
        for i in range(num):
            index = int(np.ceil(element*i))
            value = grid[index]
            values.append(value)
        return values   
    
    
    def cheby_interp(self, x, f_x, nodes=20):
        cheb_x = self.select_node(nodes, x)
        cheb_f_x = self.select_node(nodes, f_x)
        max_x = max(cheb_x)
        min_x = min(cheb_x)
        PHI = []
        for i in range(len(self.func)):
            phi = self.func[i](2*(cheb_x-min_x)/(max_x-min_x) - 1)
            PHI.append(phi)
        PHI = np.array(PHI).T
        theta = np.linalg.inv(PHI.T@PHI)@PHI.T@cheb_f_x
        return theta
        
    
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
    
    def update_chi(self, C, V):
        Vl = V[:,0]
        Vh = V[:,1]
        E_Vl = self.Tr_l[:,0]*Vl + self.Tr_l[:,1]*Vh
        E_Vh = self.Tr_h[:,0]*Vl + self.Tr_h[:,1]*Vh
        
        E_V = np.vstack((E_Vl, E_Vh))
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
        V_new = np.reshape(V_new, (len(self.grid_a),len(self.Y_grid_s)))
        ga = np.array(ga)
        ga = np.reshape(ga, (len(self.grid_a),len(self.Y_grid_s)))
        gc = np.array(gc)
        gc = np.reshape(gc, (len(self.grid_a),len(self.Y_grid_s)))
        if ret == 1:
            pos_resh = np.reshape(argm_pos,(len(self.grid_a),len(self.Y_grid_s)))
            return V_new, ga, gc, pos_resh
        elif ret == 0:
            return V_new

    
    def problem(self, start = None, Tol = 10**(-6), ret2 = 0):
        if start == None:
            V_start = np.zeros((len(self.grid_a), len(self.Y_grid_s)))
        else:
            V_start = start    
        err = 1
        j = 0
        while err>Tol:
            V_new = self.update_V(V_start, self.C)
            err = np.max(np.abs(V_start - V_new))
            V_start = V_new
            if j%100==0:
                print('   iteration value', j)
                print('   error value', err)
            j = j+1
        V_new, ga, gc, pos = self.update_V(V_start, self.C, ret = 1)
        if ret2 == 0:
            return pos, ga
        else:
            return V_new, ga, gc 
    
    def Interest_update(self, num_r = 10, r_min=0.001, r_max=0.05, maxiter=20, Tol = 0.01, pas = 0.2): #r_min can't be 0 because of the lower bound
        #r_grid = np.linspace(r_min, r_max, num_r)
        #Old_pos = np.ceil(num_r/2)
        r_up = r_max
        r_down = r_min
        r_old = (r_up+r_down)/2
        ### do something like when sign changes stop
        #Comp = 0
        j = 0
        while True:
            if j >maxiter:
                print('############# Warning ! ################')
                print('##### Maximum number of iterations #####')
                print('############# Warning ! ################')
                V_new, ga, gc = self.problem(ret2=1)
                break
            ######## when I redefine r I need to also redefine the variables that are
            ## dependent
            r = r_old
            self.r = r
            max_a = self.Y_grid_s[-1]/self.r
            if self.B==0:
                min_a = -(self.Y_grid_s[0]/self.r)*0.98
            else:
                min_a = 0
            self.K_d = ((1-self.theta)/self.r)**(1/self.theta)#because inelastic supply of L_s
            self.w = self.theta*(self.K_d)**(1-self.theta) 
            self.grid_a = np.linspace(min_a, max_a, self.N_a)
            self.Y_grid = np.tile(self.Y_grid_s, (len(self.grid_a),1)).T
            O = np.ones((len(self.Y_grid_s),len(self.grid_a)))
            self.grid_a.shape = len(self.grid_a),1
            self.mesh_a = np.kron(self.grid_a,O)
            self.mesh_Y = np.tile(self.Y_grid, (len(self.grid_a),1))
            self.grid_a.shape = len(self.grid_a),
            self.mesh_ap = np.tile(self.grid_a, (len(self.mesh_Y),1))
            self.C = self.mesh_a*(1+self.r-self.delta) + self.w*self.mesh_Y - self.mesh_ap
            #########
            
            argm_pos, ga = self.problem()
            dist = self.Inv_dist(argm_pos)
            n, c = argm_pos.shape###
            ga_endo = np.reshape(ga.T,(n*c,))##after checking reshape I decided to transpose
            Excess = dist@ga_endo
            Excess = Excess - self.K_d # for market clearing
            
            if np.abs(Excess)<Tol:
                V_new, ga, gc = self.problem(ret2=1)
                break
            
            if Excess>=0:
                r_new = pas*r_old + (1-pas)*r_down
                r_up = r_old
            elif  Excess<0:
                r_new = pas*r_old + (1-pas)*r_up
                r_down = r_old
            r_old = r_new
            
            print('iteration:',j)
            print('Excess:',Excess) 
            print('pos:',self.r)
            j = j+1    

        dist_l = dist[:self.N_a]
        theta_l = self.cheby_interp(self.grid_a, dist_l)
        dist_h = dist[self.N_a:]
        theta_h = self.cheby_interp(self.grid_a, dist_h)
        interp_dist_l = np.vectorize(lambda x: sum(theta_l[i]*self.func[i](2*(x-min_a)/(max_a-min_a) - 1) for i in range(len(self.func))))
        interp_dist_h = np.vectorize(lambda x: sum(theta_h[i]*self.func[i](2*(x-min_a)/(max_a-min_a) - 1) for i in range(len(self.func))))
            
        dist_s_l = interp_dist_l(self.grid_a)
        dist_s_l = dist_s_l + np.abs(min(dist_s_l))
        dist_s_l = dist_s_l/np.max(dist_s_l)
            
        dist_s_h = interp_dist_h(self.grid_a)
        dist_s_h = dist_s_h + np.abs(min(dist_s_h))
        dist_s_h = dist_s_h/np.max(dist_s_h)

        Structure = {}
        Structure['V'] = V_new
        Structure['ga'] = ga
        Structure['gc'] = gc
        Structure['Capital'] = self.K_d
        Structure['Saving Rate'] =  self.K_d/self.K_d**(1-self.theta)
        Structure['smoothed_dist_l'] = dist_s_l
        Structure['smoothed_dist_h'] = dist_s_h
        Structure['interest'] = r
        Structure['Excess'] = Excess
        return Structure

        
        
            
        
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        