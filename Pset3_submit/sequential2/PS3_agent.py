import math
import numpy as np
from numpy import vectorize
from scipy.optimize import fsolve

#@vectorize    #Vectorized has some trouble working with classes it didn't know that h was associated to self that is why I defined the utility function outside the class
def U(c,h,Sig,Kap,Nu):
    if c <= 0:
        u = -math.inf
    else:
        u = (c**(1-Sig))/(1-Sig) - Kap*(h**(1+1/Nu))/(1+1/Nu)
    return u

#@vectorize
def U1(c,Sig):
    if c <= 0:
        u = math.inf
    else:
        u = c**(-Sig)
    return u

#@vectorize
def U2(h, Kap, Nu):
    if h<0 or h>1:
        u = math.inf
    else:
        u = -Kap*h**(1/Nu) 
    return u

def find_zero(f,point, h=10**(-6), Tol = 10**(-3)):
    Old = point
    err=1
    while err>Tol:
        der = (f(Old+h)-f(Old-h))/((Old+h)-(Old-h))
        new = Old - f(Old)/der
        err = np.abs((new-Old)/Old)
        Old = new
    return new

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
    
    def EE(self):
        C = lambda a, h: (1-self.tho)*self.eta*h+self.y-a
        Cpos = lambda a, hp: (1-self.tho)*(self.eta+self.eps)*hp+(1+self.r)*a
        Cneg = lambda a, hp: (1-self.tho)*(self.eta-self.eps)*hp+(1+self.r)*a
        res1 = lambda a, h, hp: U1(C(a,h),self.sigma)-self.beta*(1+self.r)*(0.5*U1(Cpos(a,hp),self.sigma)+0.5*U1(Cneg(a,hp),self.sigma))
        res2 = lambda a, h, hp: (1-self.tho)*self.eta*U1(C(a,h),self.sigma)+U2(h, self.kappa, self.nu)
        EU1 = lambda a, hp: 0.5*(self.eta+self.eps)*U1(Cpos(a,hp),self.sigma)+0.5*(self.eta-self.eps)*U1(Cneg(a,hp),self.sigma)
        EU2 = lambda hp: 0.5*U2(hp, self.kappa, self.nu)+0.5*U2(hp, self.kappa, self.nu)
        res3 = lambda a, h, hp: (1-self.tho)*EU1(a,hp)+EU2(hp)
        return res1, res2, res3
    
    def update(self, start):
        res1, res2, res3 = self.EE()
        resa = lambda a: res1(a,start[1],start[2])
        resh = lambda h: res2(start[0],h,start[2])
        reshp = lambda hp: res3(start[0],start[1],hp)
        a = fsolve(resa,start[0])
        h = fsolve(resh,start[1])
        hp = fsolve(reshp,start[2])
        return a, h, hp 
    
    def problem(self, Tol=10**(-3), start = np.array([0,0.5,0.5])):
        #start = np.random.uniform(0.1,1,3) the problem came from numerical error you don't want to change starting values
        Old = start
        err = 1
        while err>Tol:
            a, h, hp = self.update(Old)
            New = np.array([a, h, hp])
            err = np.max(np.abs((New-Old)/Old))
            Old = New
        self.val = Old
        return Old
    
    def problem_expost(self):
        a = self.val[0]
        Cpos = lambda hpos: (1-self.tho)*(self.eta+self.eps)*hpos+(1+self.r)*a
        Cneg = lambda hneg: (1-self.tho)*(self.eta-self.eps)*hneg+(1+self.r)*a
        
        find_hpos = lambda hpos: (1-self.tho)*(self.eta+self.eps)*U1(Cpos(hpos),self.sigma)+U2(hpos, self.kappa, self.nu)
        find_hneg = lambda hneg: (1-self.tho)*(self.eta-self.eps)*U1(Cneg(hneg),self.sigma)+U2(hneg, self.kappa, self.nu)
        start = np.random.uniform(0,1,2)
        hpos = fsolve(find_hpos,start[0])
        hneg = fsolve(find_hneg,start[1])
        self.hpos = hpos
        self.hneg = hneg
        return hpos, hneg
    
    def wealfare(self,T1,T2):
        C = (1-self.tho)*self.eta*self.val[1]+self.y-self.val[0]+T1
        C1pos = (1-self.tho)*(self.eta+self.eps)*self.hpos+(1+self.r)*self.val[0]+T2
        C1neg = (1-self.tho)*(self.eta-self.eps)*self.hneg+(1+self.r)*self.val[0]+T2
        EU = 0.5*U(C1pos,self.hpos, self.sigma, self.kappa, self.nu)+0.5*U(C1neg,self.hneg, self.sigma, self.kappa, self.nu)
        W = U(C,self.val[1], self.sigma, self.kappa, self.nu) + self.beta*(1+self.r)*EU
        return W
    
    
            
    
    
    
    
    
    
    