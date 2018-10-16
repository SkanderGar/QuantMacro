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

class rep_ag:
    def __init__(self, theta, beta, delta, kappa, nu, kmin, kmax, hmin, hmax, num_node=20, order=3):
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
        ##### creating the basis functions
        func = []
        Psi1 = np.vectorize(lambda x: 1)
        Psi2 = np.vectorize(lambda x: x)
        func.append(Psi1)
        func.append(Psi2)
        for i in range(2,order):
            f = np.vectorize(lambda x, n=i: 2*x*func[n-1](x)-func[n-2](x))
            func.append(f)
        self.func = func
        self.gridk, self.gridk_cheb = self.cheb_node(kmin, kmax, num_node, cheby=0)
        PHI = []
        for f in self.func:
            phi = f(2*(self.gridk-self.kmin)/(self.kmax-self.kmin) -1)
            PHI.append(phi)
        self.PHI = np.array(PHI).T
        
        
        
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
    
    def update_val(self, Theta_guess, ki, start): #Theta_guess here is just for a specific ki so we also need ki
        Kp = lambda c, h: (1-self.delta)*ki + ki**(1-self.theta) *h**self.theta - c
        Kp_cheb = lambda c, h: 2*(Kp(c,h)-self.kmin)/(self.kmax-self.kmin) -1 # here the value is function of kp not k so we need to map kp to (0,1) not k
        Suma = lambda c, h: sum(Theta_guess[i]*self.func[i](Kp_cheb(c,h)) for i in range(len(self.func)))
        VnotM = lambda x: -U(x[0], x[1], self.kappa, self.nu) - self.beta*Suma(x[0],x[1]) # - the objective because I am minimizing when I want to maximize
        #non linear constraint
        const = ({'type': 'ineq', 'fun': lambda x: ki**(1-self.theta)* x[1]**self.theta -x[0]})#higher or equal to zero
        Boundc = (0.01*ki**(1-self.theta), None)
        Boundh = (0.001*self.hmin,self.hmax)
        Bound = (Boundc, Boundh)
        res = so.minimize(VnotM, start, method = 'SLSQP', bounds = Bound, constraints=const)# start should be the solution found previously so we have interest in storing previous solution 
        # it should be an enequality not an upper_bound
        Value = -res.fun
        c_opt = res.x[0]
        h_opt = res.x[1]
        return Value, c_opt, h_opt
    
    def update_theta(self, Theta_Old, Old_opt):
        New_opt = []
        V = []
        for i in range(len(self.gridk)):
            Value, c_opt, h_opt = self.update_val(Theta_Old, self.gridk[i], Old_opt[i,:]) #Old_opt is going to be a matrix containing the previews policy funtions
            New_opt.append([c_opt, h_opt])
            V.append(Value)
        New_opt = np.array(New_opt)
        V = np.array(V)
        New_theta = np.linalg.inv(self.PHI.T@self.PHI)@self.PHI.T@V
        return New_opt, New_theta
    
    def problem(self, Old_theta = None, Tol = 10**(-6)):
        if Old_theta == None:
            Old_theta = np.zeros(len(self.func))
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
        self.New_opt = New_opt
        self.New_theta = New_theta
        return New_opt, New_theta
    
    def Val_pol_fun(self):
        kc = lambda k: 2*(k-self.kmin)/(self.kmax-self.kmin) -1
        self.V = np.vectorize(lambda k: sum(self.New_theta[i]*self.func[i](kc(k)) for i in range(len(self.func))))
        
        self.Theta_c =  np.linalg.inv(self.PHI.T@self.PHI)@self.PHI.T@self.New_opt[:,0]
        self.Theta_h =  np.linalg.inv(self.PHI.T@self.PHI)@self.PHI.T@self.New_opt[:,1]
        self.gc = np.vectorize(lambda k: sum(self.Theta_c[i]*self.func[i](kc(k)) for i in range(len(self.func))))
        self.gh = np.vectorize(lambda k: sum(self.Theta_h[i]*self.func[i](kc(k)) for i in range(len(self.func))))

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        