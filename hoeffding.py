# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 11:34:00 2021

@author: McGill
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug  15 12:42:45 2021

@author: McGill

hoeffding.py

Created using python/Idle version 3.9.5 and/or Spider 5.05
For additional info on python/idle:
    https://docs.python.org/3.9/library/idle.html 
For Spyder:
    https://www.spyder-ide.org/
For plotting, I used Octave version 6.3.0
    https://www.octave.org


see also hoeffding.m

"""

#%% Imports
import numpy as np

def flip_coin(N,M):
    c = np.random.randint(2,size=(N,M))
    return c

#%% Experiments
def experiment():
    N = 1000
    M = 10
    
    c = flip_coin(N,M)
    
    c1 = 0
    
    c_rand = np.random.randint(N)
    
    heads = np.sum(c,axis=1)
    
    c_min = np.argmin(heads)
    
    # c1_flips = c[0,:]
    # c_rand_flips = c[c_rand,:]
    # c_min_flips = c[c_min,:]
    
    nu_1 = heads[c1]/M
    nu_rand = heads[c_rand]/M
    nu_min = heads[c_min]/M
    return nu_1, nu_rand, nu_min

#%% Meta Experiment Functions
def meta_experiment(bins,eps,mu):
    
    nu_1 = np.zeros(bins)
    nu_rand = np.zeros(bins)
    nu_min = np.zeros(bins)
    print("Welcome to our coin flipping Hoeffding experiment")
    print(f"Using fair coins with mu={mu}, tolerance={eps}, bins={bins}")
    Exp = 2*bins*(eps**2)
    Hoeffding_bound = float(2*np.exp(-Exp))
    print(f"Exponent {Exp:e}, so Hoeffding bound is {Hoeffding_bound:e}\n")
    for i in range(bins):
        nu_1[i], nu_rand[i], nu_min[i]= experiment()
    nu = [nu_1, nu_rand, nu_min]
    nu_mean = [nu_1.mean(),nu_rand.mean(),nu_min.mean()]
    return nu, nu_mean

def check_numueps(nu, mu, eps):
    err1 = np.abs(mu-nu[0]) - eps
    errR = np.abs(mu-nu[1]) - eps
    errm = np.abs(mu-nu[2]) - eps
    print(f"Was nu_1 within tolerance {eps}?",end='')
    if (err1 < 0):
        print(" ...Yes")
    else:
        print(" ...No")
    print(f"Was nu_r within tolerance {eps}?",end='')
    if (errR < 0):
        print(" ...Yes")
    else:
        print(" ...No")
    print(f"Was nu_min within tolerance {eps}?",end='')
    if (errm < 0):
        print(" ...Yes\n")
    else:
        print(" ...No\n")

#%% Main Program
mu = 0.5

bins = 100
eps = 0.5
print("Starting small with bins,epsilon: ", bins, eps)
nu, nu_mean = meta_experiment(bins,eps,mu)
print("Nu: ", nu_mean)
check_numueps(nu_mean,mu,eps)

bins = 100000
eps = 0.1
print("Now lets crank up the bins and decrease the tolerance to: ", bins, eps)
nu, nu_mean = meta_experiment(bins,eps,mu)
print("Nu: ", nu_mean)
check_numueps(nu_mean,mu,eps)

eps = 0.01
print("let's step down the tolerance to: ", eps)
nu, nu_mean = meta_experiment(bins,eps,mu)
print("Nu: ", nu_mean)
check_numueps(nu_mean,mu,eps)

eps = 0.001
print("step down the tolerance still further to: ", eps)
nu, nu_mean = meta_experiment(bins,eps,mu)
print("Nu: ", nu_mean)
check_numueps(nu_mean,mu,eps)

eps = 0.0001
print("last step down to: ", eps)
nu, nu_mean = meta_experiment(bins,eps,mu)
print("Nu: ", nu_mean)
check_numueps(nu_mean,mu,eps)
