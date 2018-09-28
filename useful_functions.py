import numpy as np
import math as m
from scipy.misc import derivative
import sympy as sy

def diff_uni(f, x_bar, h=10**(-6)):
    value = (f(x_bar + h)-f(x_bar - h))/((x_bar+h)-(x_bar-h))
    return value
# example:
# f = lambda x: x**2
#v = diff_uni(f,1)

def exact_diff(f, xbar,order):
    x = sy.symbols('x', real=True)
    y = f(x)
    y_prime = y.diff(x, order)
    f_prime = sy.lambdify(x, y_prime, 'numpy')
    return f_prime(xbar)  

def prod(x):
        p_old = x[0]
        if len(x)>1:
            for i in range(1,len(x)):
                p_new = p_old*x[i]
                p_old = p_new
            return p_new
        else:
            return p_old
#x = [1,2,3,4]        
#v = prod(x)

def fact(x):
    vec = list(range(1,x+1))
    value = prod(vec)
    return value
# factorial_4 = fact(4)

def diff_pol(x_bar, order, n):
    #f(x) = x^n
    coef = []
    for i in range(order):
        coef.append(n-i)
    produc = prod(coef)
    deriv = produc*x_bar**(n-order)
    return deriv

def Taylor_pol(x, x_bar, order, n):
    derivatives = []
    facts = []
    lambs = []
    for i in range(1,order+1):
              
        deriv = diff_pol(x_bar, i, n)
        derivatives.append(deriv)
        
        facto = fact(i)
        facts.append(facto)
        
        lamb = (x-x_bar)**i
        lambs.append(lamb)
    value = x_bar**n + sum(np.array(derivatives)*(np.array(lambs)/np.array(facts))) 
    return value
# v = Taylor_pol(2, 1, 3, 2) tay expansion order 3 around 1 evaluated at 2 of x^2


def Taylor_any_f(x, x_bar, order, f):
    derivatives = []
    facts = []
    lambs = []
    for i in range(1,order+1):
        if i%2==0:
            deriv = derivative(f, x_bar, n=i, order=i+1)
        else:
            deriv = derivative(f, x_bar, n=i, order=i+2)
            
        derivatives.append(deriv)
        
        facto = fact(i)
        facts.append(facto)
        
        lamb = (x-x_bar)**i
        lambs.append(lamb)
    value = f(x_bar) + sum(np.array(derivatives)*(np.array(lambs)/np.array(facts))) 
    return value

def Taylor_exact(y, x_bar, order, f):
    derivatives = []
    facts = []
    lambs = []
    for i in range(1,order+1):
        deriv = exact_diff(f, x_bar,i)
        derivatives.append(deriv)
        
        facto = fact(i)
        facts.append(facto)
        
        lamb = (y-x_bar)**i
        lambs.append(lamb)
    value = f(x_bar) + sum(np.array(derivatives)*(np.array(lambs)/np.array(facts))) 
    return value



def select_node(num, grid):
    n = len(grid)
    element = (n-1)/(num-1)# n-1 because of the problem of 100 when it shoold be 99
    values = []
    for i in range(num):
        index = int(np.ceil(element*i))
        value = grid[index]
        values.append(value)
    return values



def find_theta_mono(f,nodes):
    
    f1 = lambda x: x**5
    f2 = lambda x: x**10
    
    vnode = []
    f_tils = []
    for node in nodes:
        val = f(node)
        vnode.append(val)
        
        f_til = [f1(node), f2(node)]
        f_tils.append(f_til)
    values_f_fp = np.array([vnode])
    values_f_fp_til = np.array(f_tils).T #2 nodes
    parameter_mat = values_f_fp@values_f_fp_til.T@np.linalg.inv(values_f_fp_til@values_f_fp_til.T) #refere tp pdf
    para_v = [parameter_mat[0,0], parameter_mat[0,1]]
    return para_v

def find_theta_test(f,nodes, order):
    #In order to avoid this, you need to save the values in variables local to the lambdas, so that they don’t rely on the value of the global x:
    #lambda x, n=i: x**(n)
    func = [lambda x, n=i: x**n for i in range(order)]
    vnode = []
    f_tils = []
    for node in nodes:
        val = f(node)
        vnode.append(val)
        
        f_til = [func[i](node) for i in range(len(func))]
        f_tils.append(f_til)
    values_f_fp = np.array([vnode])
    values_f_fp_til = np.array(f_tils).T #2 nodes
    parameter_mat = values_f_fp@values_f_fp_til.T@np.linalg.inv(values_f_fp_til@values_f_fp_til.T) #refere tp pdf
    para_v = [parameter_mat[0,i] for i in range(len(func))]
    return para_v

def cheb_node(a,b,num_node):
    grid = []
    for i in range(1, num_node+1):
        cheb = m.cos((2*i-1)*m.pi/(2*num_node))
        T_cheb = (a+b)/2 + (b-a)*cheb/2
        grid.append(T_cheb)
    return np.array(grid)

def chebychev_pol(order):
    Phi0 = lambda x: 1
    Phi1 = lambda x: x
    cheb_f = [Phi0, Phi1]
    for i in range(2,order):
        func = lambda x, n=i: 2*x*cheb_f[n-1](x) - cheb_f[n-2](x)
        cheb_f.append(func)
    return cheb_f

def cheby_start_end(order, start = 0):
    vec = chebychev_pol(order)
    vec_return = vec[start:]
    return vec_return #the reason I put both of them in there is because cheb defined as a recurcive function so they have to be both in there

def find_theta_cheb(f,nodes, order, start=0):
    func = cheby_start_end(order, start= start)
    vnode = []
    f_tils = []
    for node in nodes:
        val = f(node)
        vnode.append(val)
        
        f_til = [func[i](node) for i in range(len(func))]
        f_tils.append(f_til)
    values_f_fp = np.array([vnode])
    values_f_fp_til = np.array(f_tils).T #2 nodes
    parameter_mat = values_f_fp@values_f_fp_til.T@np.linalg.inv(values_f_fp_til@values_f_fp_til.T) #refere tp pdf
    para_v = [parameter_mat[0,i] for i in range(len(func))]
    return para_v


def chebychev_pol_2dim(order, start=0):  
    Fx = cheby_start_end(order, start = start)
    Fy = cheby_start_end(order, start = start)
    pol = []
    for i in range(len(Fx)):
        for j in range(len(Fy)):
            f = lambda x, y, n=i, k=j: Fx[n](x)*Fy[k](y)
            pol.append(f)
    return pol

def find_theta_cheb_2dim(f,nodes, order, start=0):
    func = chebychev_pol_2dim(order, start=start)
    vnode = []
    f_tils = []
    for node in nodes:
        val = f(node[0], node[1])
        vnode.append(val)
        
        f_til = [func[i](node[0],node[1]) for i in range(len(func))]
        f_tils.append(f_til)
    values_f_fp = np.array([vnode])
    values_f_fp_til = np.array(f_tils).T #2 nodes
    parameter_mat = values_f_fp@values_f_fp_til.T@np.linalg.inv(values_f_fp_til@values_f_fp_til.T) #refere tp pdf
    para_v = [parameter_mat[0,i] for i in range(len(func))]
    return para_v

def Iso_quants(f, Fbar, num, Tol=10**(-1)):
    iso_k, iso_h = [], []
    grid_k = np.linspace(0, 10, num)
    grid_h = np.linspace(0, 10, num)
    for i in range(num):
        for j in range(num):
            residual = np.abs(f(grid_k[i], grid_h[j])-Fbar)
            if residual < Tol:
                iso_k.append(grid_k[i])
                iso_h.append(grid_h[j])
    return iso_k, iso_h           
            


  






        