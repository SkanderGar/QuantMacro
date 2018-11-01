import numpy as np
from scipy.stats import norm
import scipy.optimize as so
from numpy.random import uniform
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



class rep_agent:
    def __init__(self, N_a, N_s=2, rho = 0.06, Sig = 5, C_ = 1, r = 0.04, B=0, order = 3, U2 = 1):
        self.beta = 1/(1+rho)
        self.Sig = Sig
        self.C_ = C_
        self.r = r
        self.U2 = 1
        self.N_s = N_s
        self.N_a = N_a
        self.Tr, self.Y_grid = self.markov_Tr(self.N_s)
        self.max_a = self.Y_grid[-1]/self.r
        if B==1:
            self.min_a = (-self.Y_grid[0]/self.r)*0.7
        else:
            self.min_a = 0
            
        self.grid_a = np.linspace(self.min_a, self.max_a, N_a)
        self.Tr, self.Y_grid = self.markov_Tr(N_s)
        self.order=3
        
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
    
    
    def cheby_interp(self, x, f_x, nodes=10):
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
    
    def update_V(self, Theta_old, rep=0):
        V = []
        g_c = []
        theta_l = Theta_old[:,0]
        theta_h = Theta_old[:,1]
        interp_l = np.vectorize(lambda x: sum(theta_l[i]*self.func[i](2*(x-self.min_a)/(self.max_a-self.min_a) - 1) for i in range(len(self.func))))
        interp_h = np.vectorize(lambda x: sum(theta_h[i]*self.func[i](2*(x-self.min_a)/(self.max_a-self.min_a) - 1) for i in range(len(self.func))))
        if self.U2 == 1:
            for i in range(len(self.grid_a)):
                EV_l = lambda c: self.Tr[0,0]*interp_l((1+self.r)*self.grid_a[i]+self.Y_grid[0]-c) + self.Tr[0,1]*interp_h((1+self.r)*self.grid_a[i]+self.Y_grid[1]-c)
                EV_h = lambda c: self.Tr[1,0]*interp_l((1+self.r)*self.grid_a[i]+self.Y_grid[0]-c) + self.Tr[1,1]*interp_h((1+self.r)*self.grid_a[i]+self.Y_grid[1]-c)
                
                Chi_l = lambda c:  -U2(c, self.Sig) - self.beta*EV_l(c)
                Chi_h = lambda c:  -U2(c, self.Sig) - self.beta*EV_h(c)
                if (1+self.r)*self.grid_a[i] + self.Y_grid[0] - self.grid_a[-1]<=0.01:
                    Boundc_l = ((0.01, (1+self.r)*self.grid_a[i] + self.Y_grid[0] - self.grid_a[0]),)
                else:
                    Boundc_l = (((1+self.r)*self.grid_a[i] + self.Y_grid[0] - self.grid_a[-1], (1+self.r)*self.grid_a[i] + self.Y_grid[0] - self.grid_a[0]),)
                
                if (1+self.r)*self.grid_a[i] + self.Y_grid[1] - self.grid_a[-1]<=0.01:
                    Boundc_h = ((0.01, (1+self.r)*self.grid_a[i] + self.Y_grid[1] - self.grid_a[0]),)
                else:
                    Boundc_h = (((1+self.r)*self.grid_a[i] + self.Y_grid[1] - self.grid_a[-1], (1+self.r)*self.grid_a[i] + self.Y_grid[1] - self.grid_a[0]),)
                
                start_l = (0.01 + (1+self.r)*self.grid_a[i] + self.Y_grid[0] - self.grid_a[0])/2 
                start_h = (0.01 + (1+self.r)*self.grid_a[i] + self.Y_grid[1] - self.grid_a[0])/2 
                if self.iter%1==0:
                    res_l = so.minimize(Chi_l, start_l, method = 'SLSQP', bounds = Boundc_l)        
                    res_h = so.minimize(Chi_h, start_h, method = 'SLSQP', bounds = Boundc_h)        
                    v_l = -res_l.fun
                    v_h = -res_h.fun
                    ac_l = res_l.x[0]
                    ac_h = res_h.x[0]
                else:
                    v_l = -Chi_l(self.g_c[i,0])
                    v_h = -Chi_l(self.g_c[i,1])
                    ac_l = self.g_c[i,0]
                    ac_h = self.g_c[i,1]
                V.append([v_l,v_h])
                g_c.append([ac_l,ac_h])
            V = np.array(V)
            g_c = np.array(g_c)
            self.g_c = g_c
        else:
            for i in range(len(self.grid_a)):
                EV_l = lambda ap: self.Tr[0,0]*interp_l(ap) + self.Tr[0,1]*interp_h(ap)
                EV_h = lambda ap: self.Tr[1,0]*interp_l(ap) + self.Tr[1,1]*interp_h(ap)
                
                Chi_l = lambda ap:  -U1((1+self.r)*self.grid_a[i]+self.Y_grid[0]-ap, self.C_) - EV_l(ap)
                Chi_h = lambda ap:  -U1((1+self.r)*self.grid_a[i]+self.Y_grid[1]-ap, self.C_) - EV_h(ap)
                Bounda = ((self.min_a, self.max_a),)
                start = (self.min_a + self.max_a)/2
                res_l = so.minimize(Chi_l, start, method = 'SLSQP', bounds = Bounda)        
                res_h = so.minimize(Chi_h, start, method = 'SLSQP', bounds = Bounda)        
                v_l = -res_l.fun
                v_h = -res_h.fun
                V.append([v_l,v_h])
                ap_l = res_l.x[0]
                ap_h = res_h.x[0]
                g_ap.append([ap_l,ap_h])
            V = np.array(V)
            g_ap = np.array(g_ap)
        if rep == 0:
            return V
        else:
            return V, g_c
            
                
    
    def problem(self, start = None, Tol = 10**(-3), max_it=200):
        if start == None:
            theta_old = np.zeros((self.order,len(self.Y_grid)))
        else:
            theta_old = start[:]
        
        err = 1
        self.iter = 0
        while err>Tol:
            V = self.update_V(theta_old)
            theta_new_l = self.cheby_interp(self.grid_a, V[:,0])
            theta_new_h = self.cheby_interp(self.grid_a, V[:,1])
            theta_new = np.vstack((theta_new_l,theta_new_h)).T
            err = np.linalg.norm(theta_old-theta_new)/(np.linalg.norm(theta_old))
            if self.iter%10==0:
                print('iteration:',self.iter)
                print('error:',err)
            theta_old = theta_new
            self.iter = self.iter+1
            if self.iter>=max_it:
                break
        V, g_c = self.update_V(theta_old, rep=1)
        g_a = np.tile(self.grid_a, (self.N_s,1)).T*(1+self.r) + np.tile(self.Y_grid,(self.N_a,1)) - g_c
        V_interp = []
        theta_l = theta_old[:,0]
        theta_h = theta_old[:,1]
        interp_l = np.vectorize(lambda x: sum(theta_l[i]*self.func[i](2*(x-self.min_a)/(self.max_a-self.min_a) - 1) for i in range(len(self.func))))
        V_interp.append(interp_l)
        interp_h = np.vectorize(lambda x: sum(theta_h[i]*self.func[i](2*(x-self.min_a)/(self.max_a-self.min_a) - 1) for i in range(len(self.func))))
        V_interp.append(interp_h)
        
        return V_interp, g_c, g_a
    
    def simulation(self, g_c, g_a, a0 = 1, T=45, init = None):
        gc_l = lambda x: np.interp(x, self.grid_a, g_c[:,0])
        gc_h = lambda x: np.interp(x, self.grid_a, g_c[:,1])
        gc = [gc_l, gc_h]
        ga_l = lambda x: np.interp(x, self.grid_a, g_a[:,0])
        ga_h = lambda x: np.interp(x, self.grid_a, g_a[:,1])
        ga = [ga_l, ga_h]
        
        if init == None:
            start = int(1)
        #n, c = Tr.shape
        shocks = []
        shocks.append(start)
        rand = uniform(0,1,T)
        
        for i in range(T):
            prob = self.Tr[start,:]
            if rand[i] <= prob[0]:
                next_start = 0
            else:
                next_start = 1
            shocks.append(next_start)
            start = next_start
    
        shocks = np.array(shocks)
        
        C = []
        A = []
        a_old = a0
        for i in range(T):
            shock_t = shocks[i]
            a_next = ga[int(shock_t)](a_old)
            c_next = gc[int(shock_t)](a_old)
            A.append(a_next)
            C.append(c_next)
            a_old = a_next
        A = np.array(A)
        C = np.array(C)
        return A, C
        
        
        
        
        
        
        
        
        
        
        
        
        
    