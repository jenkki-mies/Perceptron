#%% Class definitions
import numpy as np

class hypothesis:
    def __init__(self, HType:str, d:int, epsilon:float, pocket, noise:float):
        # dimension
        self.dim = d
        # the type of Hypothesies: Linear vs Non vs X
        self.HType = HType
        self.w = np.zeros((1,d+1))
        self.bias = epsilon
        self.noise = noise
        self.pocket = pocket
        self.debug = False

def Phi(HType, X, plotting=False, from_lra=False):
    N = len(X)
    temp = np.ones((N,1))
    if from_lra:
        ##print("Coming from lra, X has shape:",np.shape(X))
        x,y = np.mat(X[:,0]), np.mat(X[:,1])
    else:
        sh = np.shape(X)
        ##print("Not Coming from lra, X has shape:",sh)
        x,y = np.mat(X[:,0]), np.mat(X[:,1])
        ##print("x:",np.shape(x))
        ##print("y:",np.shape(y))
    if HType == 'NL':
    ## transform non-linear data -> 1, x0^2, x1^2
        sqx = np.square(x)
        sqy = np.square(y)
        result = np.hstack((temp, sqx,sqy))
    elif HType == 'W':
        sqx = np.square(x)
        sqy = np.square(y)
        middle = np.mat(np.multiply(x,y))
        if plotting:
            print("sqx:",np.shape(sqx))
            print("sqy:",np.shape(sqy))
            print("middle:",np.shape(middle))
        result = np.hstack((temp,x,y,middle,sqx,sqy))
    else:  #default Phi, also for X.
        ## transform linear data -> 1, x0, x1
        result = np.hstack((temp, x,y))
    return result

def predict(H, data,plotting=True):
    # multiply by the weights
    phi = Phi(H.HType, data[:,[0,1]])
    if H.HType != 'W':
        w = np.array(H.w).reshape((1,3))
    else:
        w = np.array(H.w).reshape((1,6))
    phiByW = phi @  w.T
    Y = np.mat(phiByW)
    # Y needs to be an Nx1 matrix (i.e. row vector)
    return Y

def Lin_Phi(HType, X):
    N = len(X)
    temp = np.ones((N,1))
##    print("LinearPhi, X has shape:",np.shape(X))
    x,y = np.mat(X[:,0]).T, np.mat(X[:,1]).T
##    print("x:",np.shape(x))
##    print("y:",np.shape(y))
    result = np.hstack((temp, x,y))
    return result

def L_predict(H, data,plotting=False):
    if plotting:
        print("Linear Predict shapes:")
        print("Linear data:",np.shape(data))
    # multiply by the weights
    phi = Lin_Phi(H.HType, data[:,[0,1]])
    if H.HType != 'W':
        w = np.array(H.w).reshape((1,3))
    else:
        w = np.array(H.w).reshape((1,6))
    phiByW = phi @  w.T
    Y = np.mat(phiByW)
    # Y needs to be an Nx1 matrix (i.e. row vector)
    if plotting:
        print("Linear Return shape Y:", np.shape(Y))
    return Y

def Lin_Phi2(HType, X):
    N = len(X)
    temp = np.ones((N,1))
    print("LinearPhi, X has shape:",np.shape(X))
    x,y = np.mat(X[:,0]), np.mat(X[:,1])
    print("x:",np.shape(x))
    print("y:",np.shape(y))
    result = np.hstack((temp, x,y))
    return result

def L_predict2(H, data,plotting=True):
    if plotting:
        print("Linear Predict shapes:")
        print("Linear data:",np.shape(data))
    # multiply by the weights
    phi = Lin_Phi2(H.HType, data[:,[0,1]])
    if H.HType != 'W':
        w = np.array(H.w).reshape((1,3))
    else:
        w = np.array(H.w).reshape((1,6))
    phiByW = phi @  w.T
    Y = np.mat(phiByW)
    # Y needs to be an Nx1 matrix (i.e. row vector)
    if plotting:
        print("Linear Return shape Y:", np.shape(Y))
    return Y

def NL_predict(H, X,plotting=False):
    N = len(X)
    temp = np.ones((N,1))
    x,y = np.mat(X[:,0]).T, np.mat(X[:,1]).T
    sqx = np.square(x)
    sqy = np.square(y)
    if plotting:
        print("NL_Predict shapes:")
        print("temp:",np.shape(temp))
        print("X:",np.shape(X))
        print("x:",np.shape(x))
        print("y:",np.shape(y))
        print("sqx:",np.shape(sqx))
        print("sqy:",np.shape(sqy))
    phi = np.hstack((temp, sqx, sqy))
    w = np.array(H.w).reshape((1,3))
    # multiply by the weights
    phiByW = phi @  w.T
    Y = np.mat(phiByW).T
    # Y needs to be an Nx1 matrix (i.e. row vector)
    return Y

def Wtilde_predict(H, X,plotting=False):
    N = len(X)
    temp = np.ones((N,1))
    # Special case - order 2 polynomial (1, x0, x1, x0*x1, x0^2, x1^2)
    x,y = np.mat(X[:,0]).T, np.mat(X[:,1]).T
    middle = np.mat(np.multiply(x,y))
    sqx = np.square(x)
    sqy = np.square(y)
    if plotting:
        print("Wtilde_Predict shapes:")
        print("temp:",np.shape(temp))
        print("X:",np.shape(X))
        print("x:",np.shape(x))
        print("y:",np.shape(y))
        print("middle:",np.shape(middle))
        print("sqx:",np.shape(sqx))
        print("sqy:",np.shape(sqy))
    phi = np.hstack((temp,x,y,middle,sqx,sqy))
    w = np.array(H.w).reshape((1,6))
    if plotting:
        print("phi:",np.shape(phi))
        print("w:",np.shape(w))
    phiByW = phi @  w.T
    # Y needs to be an Nx1 matrix (i.e. row vector)
    Y = np.mat(phiByW).T
    return Y
