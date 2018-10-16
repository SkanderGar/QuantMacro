import numpy as np
from numpy import vectorize
from numpy.random import uniform

@vectorize
def U(c, h, kappa, nu):
    if c<=0:
        u = -np.inf
    elif c>0:
        u = np.log(c) - (kappa*h**(1+1/nu))/((1+1/nu))
    return u


class rep_agent:
    def __init__(self, theta, beta, delta, kappa, nu, kmin, kmax, hmin, hmax, n = 100, Z=[0.99,1.01]):
        self.theta = theta
        self.beta = beta
        self.delta = delta
        self.kappa = kappa
        self.nu = nu
        self.kmin = kmin
        self.kmax = kmax
        self.hmin = hmin
        self.hmax = hmax
        self.Z = Z
        self.n = n
        self.Tr = self.trans()
        Mat_O_noth = np.ones((1,n))
        Mat_O_h = np.ones((2*n,n))
        self.Mat_O_noth = Mat_O_noth
        self.Mat_O_h = Mat_O_h
        
        self.gridh = np.linspace(hmin, hmax, n)
        self.Hm = np.kron(Mat_O_h,self.gridh)
        
        gridk = np.linspace(kmin, kmax, n)
        self.gridk = gridk
        gridk_z = np.tile(gridk,len(self.Z))
        
        self.K = np.tile(gridk_z,(n,1)).T
        self.K = np.kron(self.K, Mat_O_noth)
        
        self.Kp = np.tile(gridk,(len(self.Z)*n,1))
        self.Kp = np.kron(self.Kp, Mat_O_noth)
        
        Z0 = Z[0]*np.ones((n,))
        Z1 = Z[1]*np.ones((n,))
        Zgrid = np.hstack((Z0,Z1))
        self.Zm = np.tile(Zgrid,(n,1)).T
        self.Zm = np.kron(self.Zm, Mat_O_noth)
        
        PI_lp = np.hstack((self.Tr[0,0]*np.ones((n,)),self.Tr[1,0]*np.ones((n,))))
        self.PI_lp = np.tile(PI_lp,(n,1)).T
        self.PI_lp = np.kron(self.PI_lp, Mat_O_noth) #it is multiplyin the value function you dont need it
        
        PI_hp = np.hstack((self.Tr[0,1]*np.ones((n,)),self.Tr[1,1]*np.ones((n,))))
        self.PI_hp = np.tile(PI_hp,(n,1)).T
        self.PI_hp = np.kron(self.PI_hp, Mat_O_noth)
        
    def trans(self, sigz_til = 0.01, sig_eps = 0.008):
        rho = (1-(sig_eps/sigz_til)**2)**(1/2)
        Z = [-sigz_til, sigz_til]# in this case z equals sigz
        Tr = np.empty((2,2))
        Pi = 1/2 + (rho*sig_eps**2)/(2*Z[1]**2 *(1-rho**2))
        Tr[0,0] = Pi
        Tr[0,1] = 1-Tr[0,0]
        Tr[1,0] = 1-Pi
        Tr[1,1] = 1-Tr[1,0]
        return Tr
    
    def select_node(self, num, grid):
        n = len(grid)
        element = (n-1)/(num-1)# n-1 because of the problem of 100 when it shoold be 99
        values = []
        for i in range(num):
            index = int(np.ceil(element*i))
            value = grid[index]
            values.append(value)
        return values
    
    def cheby_interp(self, grid, value, order=7, nodes=20):
        #######functions
        func = []
        Psi1 = np.vectorize(lambda x: 1)
        Psi2 = np.vectorize(lambda x: x)
        func.append(Psi1)
        func.append(Psi2)
        if order >= 2:
            for i in range(2,order):
                f = np.vectorize(lambda x, n=i: 2*x*func[n-1](x)-func[n-2](x))
                func.append(f)
        self.func = func
        
        cheb_grid = self.select_node(nodes, grid)
        cheb_grid_v = self.select_node(nodes, value)
        ma_g = max(cheb_grid)
        mi_g = min(cheb_grid)
        PHI = []
        for i in range(len(func)):
            phi = func[i](2*(cheb_grid-mi_g)/(ma_g-mi_g) -1)
            PHI.append(phi)
        PHI = np.array(PHI).T
        Theta = np.linalg.inv(PHI.T@PHI)@PHI.T@cheb_grid_v
        interp = np.vectorize(lambda x: sum(Theta[i]*func[i](2*(x-mi_g)/(ma_g-mi_g) -1) for i in range(len(func))))
        return interp
        
    def update(self, V, ret = 1):
        n, c = self.K.shape
        Vl = V[:,0]
        Vh = V[:,1]
        Vl_old = np.tile(Vl, (n,1))# think about it
        Vl_old = np.kron(Vl_old, self.Mat_O_noth)
        Vh_old = np.tile(Vh, (n,1))
        Vh_old = np.kron(Vh_old, self.Mat_O_noth)
        C = (1-self.delta)*self.K + self.Zm*self.K**(1-self.theta) *self.Hm**self.theta - self.Kp
        X = U(C, self.Hm, self.kappa, self.nu) + self.beta*(self.PI_lp*Vl_old + self.PI_hp*Vh_old)
        argm_pos_l = np.argmax(X[:int(n/2),:], axis=1)
        argm_pos_h = np.argmax(X[int(n/2):,:], axis=1)
        Vnew = []
        gk = []
        gc = []
        gl = []
        for i, idx in enumerate(list(argm_pos_l)):
            vl = X[i,idx]
            vh = X[int((n/2)+i),argm_pos_h[i]]
            gl1 = self.Kp[i,idx]
            gh1 = self.Kp[int((n/2)+i),argm_pos_h[i]]
            gl2 = C[i,idx] 
            gh2 = C[int(2*i),argm_pos_h[i]]
            gh_l2 = self.Hm[i,idx] 
            gh_h2 = self.Hm[int(2*i),argm_pos_h[i]]
            v = [vl, vh]
            g = [gl1, gh1]
            g2 = [gl2, gh2]
            g3 = [gh_l2, gh_h2]
            Vnew.append(v)
            gk.append(g)
            gc.append(g2)
            gl.append(g3)
        Vnew = np.array(Vnew)
        gk = np.array(gk)
        gc = np.array(gc)
        gl = np.array(gl)
        if ret == 1:
            return Vnew
        else:
            self.Vfl = self.cheby_interp(self.Kp[0,:],Vnew[:,0])
            self.Vfh = self.cheby_interp(self.Kp[0,:],Vnew[:,1])
            self.gk_fl = self.cheby_interp(self.Kp[0,:],gk[:,0])
            self.gk_fh = self.cheby_interp(self.Kp[0,:],gk[:,1])
            self.gc_fl = self.cheby_interp(self.Kp[0,:],gc[:,0])
            self.gc_fh = self.cheby_interp(self.Kp[0,:],gc[:,1])
            self.gl_fl = self.cheby_interp(self.Kp[0,:],gl[:,0])
            self.gl_fh = self.cheby_interp(self.Kp[0,:],gl[:,1])
            return Vnew, gk, gc, gl
    
    def problem(self, Vstart = None, Tol = 10**(-6), max_iter = 1000, type_val = 'simple'):
        if Vstart == None:
            Vstart = np.zeros((self.n,len(self.Z)))
        Vold = Vstart
        err = 1
        j = 0
        while err>Tol:
            if j%100==0:
                print('iteration:',j)
                print('error:',err)
            Vnew = self.update(Vold)
                
            err = np.max(np.abs(Vnew-Vold))
            Vold = Vnew
            j = j+1
            if j >= max_iter:
                break
        V, gk, gc, gl = self.update(Vold, ret = 0)
        return V, gk, gc, gl
    
    def simulation(self, T=100, start='low'):
        U = uniform(0,1,T)
        store_z = []
        if start == 'low':
            old_state = 0 # 0 for z lower bar, 1 for z upper bar
        elif start == 'high':
            old_state = 0
        store_z.append(old_state)
        for i in range(T):
            if U[i] < self.Tr[old_state,0]:
                new_state = 0
                store_z.append(new_state)
            elif U[i] >= self.Tr[old_state,0]:
                new_state = 1
                store_z.append(new_state)
            old_state = new_state
        self.state_z = store_z
        K = []#k will have one element more
        K.append(self.kmin)
        L = []
        C = []
        Y = []
        I = []
        W = []
        LS = []
        for i in range(T):
            if store_z[i]==0:
                Kp = self.gk_fl(K[-1])
                lc = self.gl_fl(K[-1])
                c = (1-self.delta)*K[-1] + self.Z[store_z[i]]*K[-1]**(1-self.theta) *lc**self.theta - Kp
                y = self.Z[store_z[i]]*K[-1]**(1-self.theta) *lc**self.theta
                j = Kp - (1-self.delta)*K[-1]
                w = self.theta*self.Z[store_z[i]]*K[-1]**(1-self.theta) *lc**(self.theta-1)
                ls = w*lc/y
                K.append(Kp)
                L.append(lc)
                C.append(c)
                Y.append(y)
                I.append(j)
                W.append(w)
                LS.append(ls)
                
            elif store_z[i]==1:
                Kp = self.gk_fh(K[-1])
                lc = self.gl_fh(K[-1])
                c = (1-self.delta)*K[-1] + self.Z[store_z[i]]*K[-1]**(1-self.theta) *lc**self.theta - Kp
                y = self.Z[store_z[i]]*K[-1]**(1-self.theta) *lc**self.theta
                j = Kp - (1-self.delta)*K[-1]
                w = self.theta*self.Z[store_z[i]]*K[-1]**(1-self.theta) *lc**(self.theta-1)
                ls = w*lc/y
                K.append(Kp)
                L.append(lc)
                C.append(c)
                Y.append(y)
                I.append(j)
                W.append(w)
                LS.append(ls)
        return K, C, Y, I, L, W, LS
    
    def Impulse_resp(self, Kss, T=20):
        Lss = self.gl_fl(Kss)
        Css = (1-self.delta)*Kss + Kss**(1-self.theta) *Lss**self.theta - Kss
        Yss = Kss**(1-self.theta) *Lss**self.theta
        Iss = Kss - (1-self.delta)*Kss
        Wss = self.theta*Kss**(1-self.theta) *Lss**(self.theta-1)
        K_dev = lambda k: (k-Kss)/Kss
        L_dev = lambda l: (l-Lss)/Lss
        C_dev = lambda c: (c-Css)/Css
        Y_dev = lambda y: (y-Yss)/Yss
        I_dev = lambda i: (i-Iss)/Iss
        W_dev = lambda w: (w-Wss)/Wss
        K = []#k will have one element more
        K.append(Kss)
        K_ir = []
        L_ir = []
        C_ir = []
        Y_ir = []
        I_ir = []
        W_ir = []
        Kp = self.gk_fh(Kss)
        lc = self.gl_fh(K[-1])
        c = (1-self.delta)*K[-1] + self.Z[1]*K[-1]**(1-self.theta) *lc**self.theta - Kp
        y = self.Z[1]*K[-1]**(1-self.theta) *lc**self.theta
        j = Kp - (1-self.delta)*K[-1]
        w = self.theta*self.Z[1]*K[-1]**(1-self.theta) *lc**(self.theta-1)
        
        K.append(Kp)
        K_ir.append(K_dev(Kp))
        L_ir.append(L_dev(lc))
        C_ir.append(C_dev(c))
        Y_ir.append(Y_dev(y))
        I_ir.append(I_dev(j))
        W_ir.append(W_dev(w))
        for i in range(1,T):
            Kp = self.gk_fh(K[-1])
            lc = self.gl_fh(K[-1])
            c = (1-self.delta)*K[-1] + K[-1]**(1-self.theta) *lc**self.theta - Kp
            y = K[-1]**(1-self.theta) *lc**self.theta
            j = Kp - (1-self.delta)*K[-1]
            w = self.theta*K[-1]**(1-self.theta) *lc**(self.theta-1)
            K.append(Kp)
            
            K_ir.append(K_dev(Kp))
            L_ir.append(L_dev(lc))
            C_ir.append(C_dev(c))
            Y_ir.append(Y_dev(y))
            I_ir.append(I_dev(j))
            W_ir.append(W_dev(w))
        return K_ir, L_ir, C_ir, Y_ir, I_ir, W_ir
        
    
    