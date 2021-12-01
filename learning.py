# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 12:42:45 2021

@author: McGill

learning.py

Created using python/idle version 3.9.5
For additional info on python/idle:
    https://docs.python.org/3.9/library/idle.html 
For plotting, I used Octave version 6.3.0
    https://www.octave.org

Here we are dealing with 2 learning algorithms.

Linear Regression Algorithm (LRA), uses linear algebra and
LSE to compute the best fit to data.

Perceptron Learning Algorithm (PLA), uses randomly misclassified
points to improve the predictions and iterate to a solution.

In both cases, our input data D is uniformly distributed
N x 2D points (x: -1<->+1, y: -1<->+1), that are separated
by a linear line discribed by a Target function with noise,
whose params are known only to the program, but not to the
algorithms.

The Target function formed by either

I. A line connecting 2 random points in D, and whereby for any
point in D, F(X) gives

    +1 on one side of the line,
    -1 on the other side

Or

II. A circle of radius 0.6 centered at the origin, such
that for any point in D, F(x) gives

    +1 outside of the circle,
    -1 inside the circle

Our algorithms have the goal of learning F(x) by comparing their
algorithmically determined hypothesis G(x) to known and unknown data
that will be generated iteratively.  The input data x is from a known
input distribution, while the output data y is from the combination
of the deterministic function F(x), and random noise.

Part 1a:

    D: N = 100 pts.  This is a supervised learning problem where F(x) is
        known, but F itself is unknown (to the algorithm
    1a.1 - Create a random F, and y = F(x) (training data + labels)
    1a.2 - Run the linear regression algorithm to find a g: yhat = g(x)
    1a.3 - Compare yhat and y, i.e. compute (E-in) over the training data.
    1a.4 - repeat steps 1a.1 thru 1a.3 1000 times and store g and E-in
           for later use in Part 1b.
    1a.5 - finally, we will output the average E-in score over the 1000 trials.

Part 1b: N = 100
    D: N = 1000 pts.  Same F applied to out of sample data.
    1b.1 - Create fresh data. Compute labels, y = F(x).
    1b.2 - Use g to compute yhat = g(x)
    1b.3 - Compare yhat and y, i.e. compute (E-out) over this data.
    1b.4 - repeat steps 1b.1 thru 1b.3 1000 times
    1b.5 - output the E-out average

Part 2:
    Repeat Part 1, for PLA, but starting with LRA calculated weights.

Part 3:
    Do the same experiments with Nonlinear Data,
    a. F(x) = sign(x[0]^2 + x[1]^2 - 0.6)
    b. Noise added, i.e. 10% chance that data has F(x) -> -F(x)  

Optional output files for plotting data using MATLAB/Octave:
train_data.dat
train_params.dat

see also learning.m

"""
#%% Imports
import numpy as np
from inspect import currentframe as cf
from hypothesis import hypothesis
from hypothesis import Phi
from hypothesis import predict
from hypothesis import NL_predict
from hypothesis import L_predict
from hypothesis import Wtilde_predict
from learn_plots import plot_linear_data
from learn_plots import plot_E1_vs_Eout
from learn_plots import write_data
from learn_plots import writeout_learn_data
from learn_plots import write_params

# =============================================================================
#%% Random Data Helper functions
"""
These are helper functions for generating (possibly) random data:
"""
"""
random_line      - uses 2 random points to specify a line
"""
def random_line():
    x = np.random.rand(2)*2 -1
    y = np.random.rand(2)*2 -1
    # Need to guarantee a non-infinite slope?
    while (x[0] == x[1]):
        x = np.random.rand(2)*2 -1
    m = (y[1]-y[0])/(x[1]-x[0])
    b =  y[0]-(m*x[0])
    w = line_2_weight(m, b)
    return w


def line_2_weight(m, b):
    assert b!= 0
    w = np.ones((1,3))*(-b)
    w[0,1] = -m
    w[0,2] = 1
    return w

def weight_init(F):
    w=np.zeros((1, F.dim+1))
    w[0,0] = F.bias
    return w


def set_weights(w, text=''):
    return w.copy()


def analysis(score_list, text):
    mu = np.mean(score_list)
    sigma = np.sqrt(np.var(score_list))
    if mu < .01:
        print(f"Analysis({text}): mu={mu:3.1e}, sigma={sigma:3.1e}")
    else:
        print(f"Analysis({text}): mu={mu:2.3f}, sigma={sigma:2.3f}")


"""
The main function for randomly generating data:

create_rand_data - generates rand data on [-1,+1] d-D space
                   and outputs from the random data depending on the
                   specified hyperparameters 
params: HType - 'L' or 'NL' (linear or non-linear)
        N - positive integer, number of data points
        d - dimensions, another hyperparameter
return: the Target hypothesis and data
"""
def create_rand_data(F, N, d, new_weights):
    X = np.random.rand(N,d)*2 - 1
    if new_weights and F.HType != 'W':
        # linear weights since non-linear data will be fit
        # with transformed coordinates
        w = random_line()
        w.reshape((1,3))
        # Obj Function uses these weights as params
        F.w = set_weights(w) #, f"Line {cf().f_lineno} - create_rand_data")
    if F.HType == 'X':
        F.w[0,0] = 0.6
        Y = NL_predict(F, X)
    elif F.HType == 'W':
        F.w = np.array((1,1,1,1,1,1))
        Y = Wtilde_predict(F, X)
    else:
        Y = L_predict(F, X)
    if F.noise != 0:
        noise = np.random.rand((N))-F.noise
        a = np.array(Y)
        if F.noise != 0:
            x = np.sign(noise)
        else:
            x = np.ones((N))
        b = np.mat((x)).T
        Y =  np.array(a) * np.array(b)
        Y = Y.reshape((N,1))
##    print("Creating data Shape of X:",np.shape(X))
##    print("Creating data Shape of Y:",np.shape(Y))
    data = np.hstack((X,Y))
    return data
    

# =============================================================================
#%% Algorithms helper functions

"""
randomly pick one of the errors in the errorlist
used by the PLA

"""
def pick_error(errlist):
    max = len(errlist)
    assert max != 0
    pick = np.random.randint(0,max)
    if pick > 0:
        pick = pick-1
    x = errlist[pick,:]
    return x
"""
cross validation -
    see how the learned weights score against new
    data with the specified objective Function
"""

def cross_validation(F, learned_weights, data):
    H_test = hypothesis(F.HType, F.dim, F.bias, F.pocket, F.noise)
    H_test.w = set_weights(learned_weights, f"Line {cf().f_lineno} - cv")
    correct, errs = algorithm_score(H_test, data, f"Line {cf().f_lineno}")
    score = 1.0-(correct/len(data))
    return score

"""
These are the helper functions for our algorithms

algorithm_score(H, data) - Check H against the actual data
params:
    H - the current hypothesis
    data - the current data
return:
    Number of correct data points
    an error list of data pts where H.h(pt) not equal data[:2]
"""
def algorithm_score(H, data, text):
    ErrorList = []
    N = data.shape[0]
    y = data[:,2]
    ##print("About to do an algorithm score on data with shape:",np.shape(data))
    yhat = predict(H, data) # Transpose to get a column vector (as y)
    ##print("yhat shape:",np.shape(yhat))
    ErrorIds = [ i for i in np.arange(N) if np.sign(data[i,2]) != np.sign(yhat[i,0]) ]
    ErrorList = data[ErrorIds,:2]
    correct = N - ErrorList.shape[0]
    return correct, ErrorList

def idx_of_x(data, x):
    m = len(data)
    for i in range(m):
        x1, x2 = data[i,0], data[i,1]
        if x1 == x[0,0] and x2 == x[0,1]:
            return i
    
# =============================================================================
#%% Algorithms: PLA and LRA
"""
Perceptron Learning Algorithm
"""
def PLA(T, data, M, w0, debug):
    i = 0
    correct = 0
    score = 0
    errs = []
    new_errs = []
    already_picked = []
    if debug:
        record_of_w = []
        record_of_x = []
        record_of_h = []
        record_of_correct = []
        record_of_score = []
    m = len(data)
    PLA_H = hypothesis(T.HType, T.dim, T.bias, T.pocket, T.noise)
    PLA_H.w = set_weights(w0), #f"Line {cf().f_lineno} - PLA_H.w") 
    if T.pocket:
        temp_H = hypothesis(T.HType, T.dim, T.bias, T.pocket, T.noise)
        temp_H.w = set_weights(w0)
    w = set_weights(w0)
    ##Loop over max M iters, stop early if convergence is reached
    while (i < M) and (correct < m):
        # pick a random misclassified pt
        if i != 0:
            if len(errs) == 0:
                print(f"PLA loop{i}, zero errors, correct={correct}/{m}")
            #TODO: figure out why we need to reset the weights
            elif i in [ 2, 4, 7, 11, 13, 17, 19, 23]:
                w = set_weights(w0)
            x = pick_error(errs)            
        else:
            x = np.array(data[0,:2])
        x = np.array((x)).reshape((1,2))
##        j = idx_of_x(data, x)
##        if j in already_picked:
##            #print(f"i={i},random j already picked (len errlist={len(errs)}) : data row {j}/{m}: {data[j,:2]}")
##            w = set_weights(w0)
##            correct = 0
##            errs = []
##        else:
##            #print(f"i={i},random j newly picked (len errlist={len(errs)}) : data row {j}/{m}: {data[j,:2]}")
##            already_picked.append(j)
        if T.pocket:
            sign = predict(temp_H, x)
        else:
            sign = predict(PLA_H, x)
        ## can't use np.sign, because 0 will get us stuck
        if sign >= 0:
            dy_ = Phi(T.HType,x)
        else:
            dy_ = -Phi(T.HType,x)
        dy = dy_.reshape((1,3))
        #instead of scaling to dy, only use +1 or -1
        w -= np.sign(dy)
#        w -= dy
        if T.pocket:
            temp_H.w = set_weights(w)
            new_correct, new_errs = algorithm_score(temp_H, data, f"Line {cf().f_lineno}, T.w={T.w}")
            # only update the hypothesis if it scores better
            if new_correct > correct:
                if debug:
                    print(f"Improving on LRA: {correct}->{new_correct}")
                #adopt the new hypothesis
                PLA_H.w = set_weights(temp_H.w, f"Line {cf().f_lineno} taking temp_H.w")
                correct = new_correct
                errs = new_errs
            elif i == 0:
                correct = new_correct
                errs = new_errs
        else:
            PLA_H.w = set_weights(w, f"Line {cf().f_lineno}")
            ### IMPORTANT LINE OF CODE below:
            correct, errs = algorithm_score(PLA_H, data, f"Line {cf().f_lineno}, T.w={T.w}")          
        score = 1.0-(correct/m)
        if debug:
            record_of_w.append(PLA_H.w)
            record_of_x.append(x)
            record_of_h.append(sign)
            record_of_correct.append(correct)
            record_of_score.append(score)
        i += 1 # while loop iteration
    if debug:
        print("outputting learn to file...")
        writeout_learn_data(T, i, record_of_w, record_of_x, record_of_h, record_of_correct, record_of_score)
    return i+1, PLA_H.w, score
# =============================================================================
"""
# Linear Regression Algorithm
# "One-shot learning"
#
"""
def LRA(T, data, debug=False):
    if T.HType == 'X':
        pcorrect, perrs = algorithm_score(T, data, f"Line {cf().f_lineno}")
    N = len(data)
    # data contains original X, but Ys includes noise
    X = data[:,:T.dim]
    if debug:
        print("LRA: pre Phi shape of X...", X.shape)
    X = Phi(T.HType, X, plotting=True, from_lra=True)
    if debug:
        print("LRA: post Phi shape of X...", X.shape)
        print("LRA X=", X)
    
    Y = data[:,[T.dim]] # Originally, I used the linear not logistic value
##    Y = np.sign(data[:,[T.dim]]) # Just use the last column
    if debug:
        print("LRA: Y used for LRA=", Y)
    w = np.transpose((np.linalg.pinv(X.T@X)@ X.T) @ Y)
    if debug:
        print("LRA: shape of w... double checking", w.shape)
        print("LRA w=", w)
    
    LRA_H = hypothesis(T.HType, T.dim, T.bias, T.pocket, T.noise)
    LRA_H.w = set_weights(w, f"Line {cf().f_lineno} - LRA")
    correct, errs = algorithm_score(LRA_H, data, f"Line {cf().f_lineno}")
    if T.HType == 'X':
        if correct <= pcorrect:
            LRA_H.w = T.w.copy()
            correct = pcorrect
    score = 1.0-(correct/N)
    return LRA_H.w, score
# =============================================================================
#%% Main Loop
"""
main_loop:
  Evaluate learning algorithms according to the hyperparameter
  specifications, generating both in-sample and out-sample data,
  measuring Ein and Eout, and storing the results
          use_lra, noise, Data_f, Param_f, seed, debug)

# params: all the hyperparameters
# return: None
#
"""
def main_loop(HType, dim, pocket, epsilon, N_in, N_out, pla_max, iters,
              use_lra, do_pla, noise, Data_f, Param_f, seed, debug_list):
    
    # List Initializations
    data = []
    a1_scores_Ein = []
    a2_scores_Ein = []
    a1_scores_Eout = []
    a2_scores_Eout = []
    c_scores = []
    TargetWeights = []
    FinalPLAWeights = []
    FinalPocketWeights = []
    FinalLRAWeights = []

    if seed != None:
        np.random.seed(seed)
    elif seed != 39:
        seed = np.random.randn(iters)

    F = hypothesis(HType, dim, epsilon, pocket, noise)

    # =============================================================================
    # Part 1a: compute weights, E_in for Perceptron, LRA
    # =============================================================================
    for idx in range(iters): # 1a.4/1.b4 repeat iteration times
        # 1a.1 Create a random Target Function, F
        # 1a.2 Create N_in random data points for training
        train_data = create_rand_data(F, N_in, dim, True)
        # since I'm looping the experiment a number of times
        # in order to get performance statistics, I will only
        # do debug if idx is in the debug_list
        debug = idx in debug_list
            
        # 1a.3 supervised learning via train_data[:,dim+1]
        if (use_lra):
            # another hyperparam - LRA_LIM should be ~ 10000
            # Don't run linear regression if there's too much data
            # for the pinv matrix calculations
            lra_g, e_in_lra = LRA(F, train_data, debug)
            w = set_weights(lra_g, f"Line {cf().f_lineno} - lra_g") # start perceptron with lse weights
        if do_pla:
            #No not using lra because it takes too long
            if use_lra:
                if debug:
                    print(f"PLA with Linear Regression start weight, w=",w)
            else:
                w=weight_init(F)
                if debug:
                    print(f"PLA without Linear Regression, w=",w)
            c,pla_g,e_in_pla = PLA(F, train_data, pla_max, w, debug)
    # =============================================================================
    # Part 1b1-3: Cross Validation, i.e. measure E_out
    # =============================================================================
        TargetWeights.append(F.w)
        cvdata = create_rand_data(F, N_out, F.dim, False)
        if use_lra:
            e_out_lra = cross_validation(F, lra_g, cvdata)
            if do_pla:
                e_out_pla = cross_validation(F, pla_g, cvdata)
            FinalLRAWeights.append(lra_g)
            a1_scores_Ein.append(e_in_lra)
            a1_scores_Eout.append(e_out_lra)
        if do_pla:
            e_out_pla = cross_validation(F, pla_g, cvdata)
            # 1a.4/1.b4 Store the results
            FinalPLAWeights.append(pla_g)
            a2_scores_Ein.append(e_in_pla)
            a2_scores_Eout.append(e_out_pla)
            c_scores.append(c)
        data.append(train_data)
        if (debug):
            h = F
            if HType == 'W':
                h.w = lra_g
            fig, ax = plot_linear_data(h, train_data)
            if do_pla:
                h.w = pla_g
                plot_linear_data(h, train_data, pla=True, fig=fig, ax1=ax)
            if use_lra:
                tH = F.w.copy()
                h.w = lra_g
                plot_linear_data(h, train_data, lra=True, fig=fig, ax1=ax,tempH=tH)
                
    # ===================================================================
    # write the experiment results to data/param files for analysis
    # ===================================================================
    if do_pla:
        hparams = [HType, dim, N_in, N_out, pocket, iters, pla_max]
        results = [TargetWeights, FinalPLAWeights, FinalLRAWeights,
                   a1_scores_Ein, a2_scores_Ein,
                   a1_scores_Eout,a2_scores_Eout,
                   c_scores]
        write_params(Param_f, hparams, results)
        file = open(Data_f, 'w', newline='')
        for idx in debug_list:
            file.write(f"Train Data {idx} with shape:{np.shape(data[idx])}\n")
            F.w = set_weights(FinalPLAWeights[idx], f"Line {cf().f_lineno}")
            Y_hat = predict(F, data[idx])
            ##write_data(file, data[idx], Y_hat, extra=True)
        file.close()
        plot_E1_vs_Eout(a2_scores_Ein,a2_scores_Eout,f"{F.HType}-{F.dim}d Ein vs Eout noise={F.noise}")

    # =============================================================================
    # Part 1a.5/1b.5 : Output the averages Ein vs Eout
    # =============================================================================
    if use_lra:
        print(f"Linear Regression Algorithm performance:")
        analysis(a1_scores_Ein, 'LRA_Ein')
        analysis(a1_scores_Eout,'LRA_Eout')
        #print("Full listing of LRA Ein - a1_scores_Ein:", a1_scores_Ein)

    if do_pla:
        print(f"Perceptron Learning Algorithm performance:")
        analysis(a2_scores_Ein, 'PLA_Ein')
        analysis(a2_scores_Eout,'PLA_Eout')
        analysis(c_scores,'PLA_iters')
        #print("Full listing of PLA iters:", c_scores)
        if np.mean(a2_scores_Ein) > 0:
            idx = np.argmax(a2_scores_Ein)
            val = a2_scores_Ein[idx]
            print(f"PLA was non-zero on trial {idx} - Ein={val}!!")
        if np.mean(a2_scores_Eout) > 0:
            idx = np.argmax(a2_scores_Eout)
            val = a2_scores_Eout[idx]     
            print(f"PLA was non-zero on trial {idx} - Eout={val}!!")
        if np.mean(c_scores) > 0:
            idx = np.argmax(c_scores)
            val = c_scores[idx]
            print(f"PLA was non-zero on trial {idx} - c={val}!!")



# =============================================================================
#%% Main()
"""
Program starts here - takes the hyperparameters and calls the main_loop

TODO:
    -could start by reading the hyperparameters from a file
    -check if octave plotting still works,
    -implement Python based plotting
    -divide the data into predicted and actual
    -add GUI user interface widgets
    -hardcoded dimensions - w0 array indices, train_data indices
main_loop(HType, dim, pocket, epsilon, N_in, N_out, pla_max, iters,
              octave, Data_f, Param_f, noise, seed, debug)
"""    
# Hyperparameter Initializations
dim     = 2
# =============================================================================
#  HW2 problem 5:
##N_in    = 100
##N_out   = 10
##pocket  = False #don't care about PLA in problem 5
##HType = 'L'
##do_pla = False
##noise = 0.0 #for Non-linear problem, use noise = 0.1
# =============================================================================
# =============================================================================
#  HW2 problem 7:
N_in    = 10
N_out   = 10
pocket  = True
do_pla = True
HType = 'L'
noise = 0.0 #for Non-linear problem, use noise = 0.1
# =============================================================================
#  HW2 problem 8:
##N_in    = 1000
##N_out   = 100
##pocket  = True
##do_pla = True
##linear  = False
##transform = False
##noise = 0.1 #for Non-linear problem, use noise = 0.1
##Results: Analysis(PLA_iters): mu=282.560, sigma=448.027
# =============================================================================
##    HType = 'L'
##    HType = 'NL'
##    HType = 'X'
##    HType = 'W'
epsilon = 0.001
iters   = 1000 # overall iters of experiment for final analysis
#whether to use lra depends on:
#the number of data points, N_in and iterations
#since we have to perform expensive matrix calculations per iteration,
#which scale exponentially with N_in and linearly with iterations
LRA_LIM = 1e6
use_lra = (iters*N_in <= LRA_LIM)
pla_max = 1000 # max number of iters for PLA
seed    = 40
debug_list   = [seed]
Data_f  = "octave/train_data.dat"
Param_f = "octave/train_params.dat"

# =============================================================================
# Start by printing the Hyperparameters
# =============================================================================
print(f"Perceptron learning: {HType}{dim}-d with {noise*100}% noise")
print(f"pocket={pocket},use_lra={use_lra} N_in={N_in},N_out={N_out}")
print(f"Runs={iters},Max for PLA={pla_max},seed={seed}")

# Now start the simulation
main_loop(HType, dim, pocket, epsilon, N_in, N_out, pla_max, iters,
          use_lra, do_pla, noise, Data_f, Param_f, seed, debug_list)
print("Learning complete.")
