# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 12:42:45 2021

@author: McGill

perceptron.py

Created using python/idle version 3.9.5
For additional info on python/idle:
    https://docs.python.org/3.9/library/idle.html 
For plotting, I used Octave version 6.3.0
    https://www.octave.org

Perceptron Learning Algorithm (PLA) with
N x 2D points,

(x: -1<->+1, y: -1<->+1)
and a randomly generated Target function(line)
created by forming the line connecting 2 other
such random 2D points, whereby F(X) gives
    +1 on side of this line,
    -1 on the other side
The goal of the algorithm is to learn F by applying
the PLA.

Part 1: N = 10
Part 2: N = 100

For each part, after learning the line, the resulting
hypothesis is tested through new validation data.
The validation data set has 1000 x 2-D points.

In each part of this code, PLA is iterated 1000 times
to come up with a final hypothesis, and the entire
experiment is repeated 100 times with new training and
validation data sets for each run.  This is done
in order to experimentally compute a mean and variance
for convergence (c) and generalization (g) scores that are
used for analyzing the algorithm performance.

Convergence is the number of iterations necessary for the
final hypothesis to match the target function on
all N training data points.

While generalization is measured by applying the final hypothesis
to new validation data that is randomly created post-training.

In other words, we have 2 data sets, training and validation.
We could also further subdivide the training data into training
and cross validation data, but that is unimplemented here.
So, statistically speaking, the training data is our "in-sample",
dataset, while the validation data is the out-sample dataset.

Optional output files for plotting using MATLAB/Octave:
train_data.dat
train_params.dat
train_data2.dat
train_params2.dat

see also perceptron.m

"""

#%% Imports
import numpy as np

#%% Class definitions

class hypothesis_2d:
    def __init__(self,m=1,b=1):
        assert b != 0
        self.w= np.array((-m/b, 1.0/b))
    def updateWeights(self, w):
        self.w = w.copy()
    def h(self, pt):
        return np.sign(np.dot(self.w, pt))
"""
These are functions used for randomly generating data:

create_random - generates N random points on [-1,+1 : -1,+1] d-D space 
random_line - uses random coordinates to create a line
"""
#%% Random data Helper functions
def create_random(N, d):
    # x,y coordinates for 2 pts in [-1,1]
    return np.random.rand(N,d)*2 -1


def random_line():
    x = create_random(2,2)
    while x[0,0] == x[1,0]:
        x = create_random_2d(2)
    assert x[0,0] != x[1,0]
    m = (x[1,1] - x[0,1])/(x[1,0] - x[0,0])
    b = x[0,1] - (m*x[0,0])
    return m, b

        

"""
These are the functions for a single run of the perceptron algorithm
on a training data set:

perceptron_score - returns a Python list of mismatches for a hypotheses, assuming
                   the specified Target
                 - scores the hypothesis, assuming the specified Target, returning
                   the integer count of correct classifications
PLA - one run of the PLA for a given error list and hypothesis
      returns a newly computed weight vector
run_validation - uses a validation data set to score a hypothesis given a Target
run_perceptron - iterates PLA on the Target/training data, M times

"""
#%% perceptron validation/learning and Perceptron Hypothesis Checks
def perceptron_score(H1, H2, data,debug=False):
    ErrorList = []
    correct = 0
    assert data.shape[0] > 0
    for i in range(data.shape[0]):
        p = np.array((data[i,0],data[i,1]))
        data[i,2] = np.sign(H1.h(p))
        if H1.h(p) != H2.h(p):
            ErrorList.append(p)
        else:
            correct += 1
    return correct, ErrorList

def PLA(errlist, H):
    w = H.w.copy()
    max = len(errlist)
    if max != 0:
        pick = np.random.randint(0,max)
        x = errlist[pick]
        w += x*(-H.h(x))
    return w

def run_validation(Target, g, N):
    # Create N_val random data points for validation
    validation_data = create_random(N,3)
    G = hypothesis_2d()
    G.updateWeights(g)
    correct, errs = perceptron_score(Target, G, validation_data)
    m = len(validation_data)
    assert m != 0
    assert m == correct + len(errs)
    return 1.0-(correct/m)

def run_perceptron(Target, train_data, M=1000, debug=False):
    #starting with weights all 0
    H = hypothesis_2d()
    i = 0
    correct, errs = perceptron_score(Target, H, train_data)
    ##Loop over max M iterations, stop early if convergence is reached
    while (i < M) and (correct != len(train_data)): 
        w = PLA(errs, H) # pick a random misclassified pt
        H.updateWeights(w)  # and compute/update new weights
        correct, errs = perceptron_score(Target, H, train_data, debug)
        i += 1 # while loop iteration
    return i, H.w

"""
write_data
write_params
write_matlab_octave_files

helper functions for writing data/params to file:
Only needed if plotting the data with Octav3e (see perceptron.m) 

"""
#%% File Output
def write_data(outfile, L):
    assert(L.shape[0] > 0)
    for i in range(L.shape[0]):
        assert(L.shape[1] == 3)
        outputLine = f"{L[i,0]},{L[i,1]},{L[i,2]}\n"
        outfile.write(outputLine)

def write_params(outfile, w1, w2, N, c_score, g_score, M_iter):
    assert(w1[1] != 0)
    assert(w2[1] != 0)
    # compute slope, intercept of the lines defined by the weights
    m1 = -w1[0]/w1[1]
    m2 = -w2[0]/w2[1]
    b1 = 1/w1[1]
    b2 = 1/w2[1]
    text = f"{m1},{b1},{m2},{b2},{N:d},{c_score},{g_score},{M_iter:d}\n"
    outfile.write(text)

def write_matlab_octave_files(Data_f, Param_f, TargetWeights, FinalHWeights,
                              N, train_data, c_score, g_score, M_iter):
    m = len(TargetWeights) # m=number of runs/iterations of the experiment
    assert (len(FinalHWeights) == m)
    assert (len(train_data) == m)
    assert (len(c_score) == m)
    assert (len(g_score) == m)
    file1 = open(Data_f, 'w', newline='')
    file2 = open(Param_f, 'w', newline='')
    for i in range(m):
        w1 = TargetWeights[i]
        w2 = FinalHWeights[i]
        data = train_data[i]
        write_params(file2, w1, w2, N, c_score[i], g_score[i], M_iter)
        write_data(file1, data)
    file1.close()
    file2.close()
# =============================================================================

#%% Main()

#  Initializations
N_train1 = 10
N_train2 = 100
N_validation_pts = 1000
M_iter = 1000
M_runs = 100 # number of runs of the experiment
octave = True
# Use a seed, only if comparing results on different systems
seed = 1234
#np.random.seed(seed)
Data_f1 = "octave/train_data1.dat"
Param_f1 = "octave/train_params1.dat"
Data_f2 = "octave/train_data2.dat"
Param_f2 = "octave/train_params2.dat"

# =============================================================================
# Part 1: N = 10
# =============================================================================
print(f"Running Perceptron Part1(N={N_train1},M={M_iter}, seed={seed})")
c_scores = []
g_scores = []
processed_data = []
TargetWeights = []
FinalHWeights = []
for idx in range(M_runs):
     # Create a random Target Function, F
     slope, intercept = random_line()
     F = hypothesis_2d(slope, intercept)
     # Create N_train1 random data points for training
     train_data = create_random(N_train1,3) ##using z for the sign +/-
     # Train over data and F
     debug = idx in (0,1,2)
     c, G = run_perceptron(F, train_data, M=M_iter,debug=debug)
     # Store the results for later analysis
     c_scores.append(c)
     TargetWeights.append(F.w)
     FinalHWeights.append(G)
     processed_data.append(train_data)
     # Validation or Outsample 
     g = run_validation(F, G, N_validation_pts)
     g_scores.append(g)
# =============================================================================
# 
# write the data to data/param files for plotting with MATLAB/Octave
#
# =============================================================================
if octave:
    write_matlab_octave_files(Data_f1, Param_f1, TargetWeights, FinalHWeights,
                              N_train1, processed_data, c_scores,g_scores, M_iter)
print(f"Perceptron part1 M({M_runs}runs/{M_iter}iterations) complete")
mu_c = np.mean(c_scores)
sigma_c = np.sqrt(np.var(c_scores))
print(f"mu, sigma(c) = {mu_c}, {sigma_c}:")
print(c_scores)
mu = np.mean(g_scores)
sigma = np.sqrt(np.var(g_scores))
print(f"P(Error) = {mu}, {sigma}:")
print(f"Accuracy = {100*(1-mu)} +/- {sigma}%")
# =============================================================================
# Part 2: N = 100
# =============================================================================
print(f"Running Perceptron Part2(N={N_train2},M={M_iter}, seed={seed})")
c_scores = []
g_scores = []
processed_data = []
TargetWeights = []
FinalHWeights = []
for idx in range(M_runs):
     # This time we have N_train2 points
     slope, intercept = random_line()
     F = hypothesis_2d(slope, intercept)
     train_data = create_random(N_train2,3)
     c, G = run_perceptron(F, train_data, M=M_iter)
     # Store the results for later analysis
     c_scores.append(c)
     TargetWeights.append(F.w)
     FinalHWeights.append(G)
     processed_data.append(train_data)
     g = run_validation(F, G, N_validation_pts)
     g_scores.append(g)
# =============================================================================
# 
# write the data to data/param files for plotting with MATLAB/Octave
#
# =============================================================================
if octave:
    write_matlab_octave_files(Data_f2, Param_f2, TargetWeights, FinalHWeights,
                              N_train2, processed_data, c_scores, g_scores, M_iter)
print(f"Perceptron part2 M({M_runs}runs/{M_iter}iterations) complete")
mu_c = np.mean(c_scores)
sigma_c = np.sqrt(np.var(c_scores))
print(f"mu, sigma(c) = {mu_c}, {sigma_c}:")
print(c_scores)
mu = np.mean(g_scores)
sigma = np.sqrt(np.var(g_scores))
print(f"P(Error) = {mu}, {sigma}:")
print(f"Accuracy = {100*(1-mu)} +/- {sigma}%")
