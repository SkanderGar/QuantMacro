import numpy as np
from numpy import vectorize
import scipy.optimize as so

@vectorize
def U(c, h, kappa, nu):
    if c<=0:
        u = -np.inf
    elif c>0:
        u = np.log(c) - (kappa*h**(1+1/nu))/((1+1/nu))
    return u

class Rep_age_shock:
    def __init__(self, theta, beta, delta, kappa, nu, kmin, kmax, hmin, hmax, Z = [0.99, 1.01] , num_node=20, order=3):
        self.theta = theta
        self.beta = beta
        self.delta = delta
        self.kappa = kappa
        self.nu = nu
        self.kmin = kmin
        self.kmax = kmax
        self.hmin = hmin
        self.hmax = hmax
        self.num_node = num_node
        self.order = order
        self.Z = Z
        ##### creating the basis functions
        func = []
        Psi1 = lambda x: 1
        Psi2 = lambda x: x
        func.append(Psi1)
        func.append(Psi2)
        if order >= 2:
            for i in range(2,order):
                f = lambda x, n=i: 2*x*func[n-1](x)-func[n-2](x)
                func.append(f)
        self.func = func
        func2d = []
        for i in range(len(func)):
            for j in range(len(func)):
                f = np.vectorize(lambda k, z, n=i: func[n](k)*func[n](z))
                func2d.append(f)
        self.func2d = func2d
        gridk = self.cheb_node(kmin, kmax, num_node)
        zm = self.Z[0]*np.ones(len(gridk))
        zp = self.Z[1]*np.ones(len(gridk))
        self.gridk = np.hstack((gridk,gridk))
        self.Zgrid = np.hstack((zm,zp))
        self.gridk_cheb = 2*(self.gridk-self.kmin)/(self.kmax-self.kmin) -1
        #self.Zgrid_cheb = 2*(self.Zgrid-self.Z[0])/(self.Z[1]-self.Z[0]) -1
        PHI = []#check if this works
        for f in self.func2d:
            Phi = f(self.gridk_cheb,self.Zgrid)
            PHI.append(Phi)
        PHI = np.array(PHI).T
        self.PHI = PHI
        self.Tr = self.trans()
        self.prob_Lb = np.tile(self.Tr[0,:],(len(self.gridk),1))
        self.prob_Ub = np.tile(self.Tr[1,:],(len(self.gridk),1))
        
    def cheb_node(self, a, b, num, cheby=1):
        vec = np.arange(0,num)
        vec = np.flip(vec, axis=0)
        chb = np.cos((vec*np.pi)/(num-1))
        points = (a+b)/2 + ((b-a)/2)*chb
        if cheby == 0:
            vec_unit = 1/2 + (1/2)*chb
            return np.array(points), np.array(vec_unit)
        else:
            return np.array(points)
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
    
    def update_val(self, Theta_guess, ki, zi, start): #Theta_guess here is just for a specific ki so we also need ki

        ###########LB_UB############
        Kp = lambda c, h: (1-self.delta)*ki + zi*ki**(1-self.theta) *h**self.theta - c
        #Kp_cheb = lambda c, h: 2*(Kp(c,h)-self.kmin)/(self.kmax-self.kmin) -1 # here the value is function of kp not k so we need to map kp to (0,1) not k
        if zi == self.Z[0]:
            idx = 0
        elif zi == self.Z[1]:
            idx = 1
        Kp_cheb = lambda c, h: 2*(Kp(c,h)-self.kmin)/(self.kmax-self.kmin) -1
        #Z_cheb = lambda z: 2(z-self.Z[0])/(self.Z[1]-self.Z[0]) -1
        Suma = lambda c, h, z: sum(Theta_guess[i]*self.func2d[i](Kp_cheb(c,h),z) for i in range(len(self.func2d)))
        VnotM = lambda x: -U(x[0], x[1], self.kappa, self.nu) - self.beta*(self.Tr[idx,0]*Suma(x[0],x[1],self.Z[0]) + self.Tr[idx,1]*Suma(x[0],x[1],self.Z[1])) # instead of self.Z[1] we have 1 cause converted in cheb element- the objective because I am minimizing when I want to maximize
        #non linear constraint
        const = ({'type': 'ineq', 'fun': lambda x: zi*ki**(1-self.theta)* x[1]**self.theta -x[0]})#higher or equal to zero
        Boundc = (0.01*zi*ki**(1-self.theta), None)
        Boundh = (0.001*self.hmin,self.hmax)
        Bound = (Boundc, Boundh)
        res = so.minimize(VnotM, start, method = 'SLSQP', bounds = Bound, constraints=const)# start should be the solution found previously so we have interest in storing previous solution 
        
        Value = -res.fun
        c_opt = res.x[0]
        h_opt = res.x[1]
        return Value, c_opt, h_opt
    
    def update_theta(self, Theta_Old, Old_opt):
        New_opt = []
        V = []
        for i in range(len(self.gridk)):
            Value, c_opt, h_opt = self.update_val(Theta_Old, self.gridk[i], self.Zgrid[i], Old_opt[i,:]) #Old_opt is going to be a matrix containing the previews policy funtions
            New_opt.append([c_opt, h_opt])
            V.append(Value)
        New_opt = np.array(New_opt)
        V = np.array(V)
        New_theta = np.linalg.inv(self.PHI.T@self.PHI)@self.PHI.T@V
        New_theta = np.array(New_theta)
        return New_opt, New_theta
    
    def problem(self, Old_theta = None, Tol = 10**(-6)):
        if Old_theta == None:
            Old_theta = np.zeros(len(self.func2d))
            Old_c = (self.kmax/4)**(1-self.theta) *np.ones(len(self.gridk))
            Old_h = (self.hmax/4)*np.ones(len(self.gridk))
            Old_opt = np.vstack((Old_c,Old_h)).T
        err = 1
        j = 0
        while err>Tol:
            New_opt, New_theta = self.update_theta(Old_theta, Old_opt)
            err = np.max(np.abs(Old_theta-New_theta))
            if j%50 == 0:
                print('iteration:', j)
                print('error:', err)
            Old_theta = New_theta
            Old_opt = New_opt
            j = j+1
        return New_opt, New_theta