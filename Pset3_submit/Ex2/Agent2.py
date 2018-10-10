import math
import numpy as np
from numpy import vectorize

@vectorize    #Vectorized has some trouble working with classes it didn't know that h was associated to self that is why I defined the utility function outside the class
def U(c,h,Sig,Kap,Nu):
    if c <= 0:
        u = -math.inf
    else:
        u = (c**(1-Sig))/(1-Sig) - Kap*(h**(1+1/Nu))/(1+1/Nu)
    return u

class Agent:
    def __init__(self, y, eta, eps, r, tho=0, beta=0.99, nu=4, sigma=3, kappa = 4):
        self.y = y
        self.eta = eta
        self.nu = nu
        self.sigma = sigma
        self.kappa = kappa
        self.eps = eps
        self.beta = beta
        self.tho = tho
        self.r = r
    
    
    def find(self,Ma, Mh, Mh_p):
        T1 = self.tho
        T2 = self.tho
        V = U((1-self.tho)*self.eta*Mh+self.y+T1-Ma, Mh, self.sigma, self.kappa, self.nu)+\
                    self.beta*(0.5*U((1-self.tho)*(self.eta+self.eps)*Mh_p+(1+self.r)*Ma+T2, Mh_p, self.sigma, self.kappa, self.nu)+\
                               0.5*U((1-self.tho)*(self.eta-self.eps)*Mh_p+(1+self.r)*Ma+T2, Mh_p, self.sigma, self.kappa, self.nu))
        max_idx = np.argmax(V) 
        max_idx = np.unravel_index(max_idx, np.array(V).shape)
        
        #period 2 given optimal choice of asset in 1 what do I do in 2
        V1_p = U((1-self.tho)*(self.eta+self.eps)*Mh[:,0]+(1+self.r)*Ma[max_idx[0],0]+T2, Mh[:,0], self.sigma, self.kappa, self.nu)
        max_idx1 = np.argmax(V1_p) 
        
        V2_p = U((1-self.tho)*(self.eta-self.eps)*Mh[:,0]+(1+self.r)*Ma[max_idx[0],0]+T2, Mh[:,0], self.sigma, self.kappa, self.nu)
        max_idx2 = np.argmax(V2_p)
        
        h_ex_post = (max_idx1,max_idx2)
        
        return V[max_idx], max_idx, h_ex_post