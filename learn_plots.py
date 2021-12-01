import numpy as np
import matplotlib.pyplot as plt
from hypothesis import hypothesis
from hypothesis import Phi
from hypothesis import L_predict
from hypothesis import L_predict2
from hypothesis import NL_predict
from hypothesis import Wtilde_predict

def weight_2_circle(w, angle):
    r0 = np.sqrt(np.abs(w[0,0]))
    rx = r0 * np.cos(angle) # still need to implement offset
    ry = r0 * np.sin(angle) # still need to implement offset
    return r0, rx, ry

def weight_2_line(w, x,bias=0):
    if (w[0,1] == 0):
        print("horizontal")
        #return -1
    if (w[0,2] == 0):
        print("vertical")
        return -1
    m = -w[0,1]/w[0,2]
    b = -w[0,0]/w[0,2]
    y = m*x + b
    return m, b, y

"""
Need to add Y to the X
"""
def plot_linear_data(F, data,pla=False,lra=False,fig=0,ax1=0,tempH=None):
    xrange      = np.arange(-1, 1, 0.01)
    if F.HType != 'W':
        m1, b1, yrange  = weight_2_line(F.w, xrange,bias=F.bias)
    if fig== 0: ##not pla and not lra or nonlinear:
        N           = data.shape[0]
        y           = data[:,2]
        if F.HType == 'L':
            print("plot_linear with HType L")
            d = data[:,:2]
            yhat        = L_predict2(F, d, plotting=True)
            print("yhat Shape:",np.shape(yhat))
        elif F.HType == 'NL':
            print("plot_linear with HType NL")
            yhat        = NL_predict(F, data, plotting=True).T
        elif F.HType == 'W':
            print("plot_linear with HType W")
        elif F.HType == 'X':
            print("plot_linear with HType X")
            ##Note that we are using a linear prediction on non-linear data in this case 
            yhat        = L_predict(F, data, plotting=True).T
        ## index array for list comprehensions with filters
        idAll       = np.arange(N)
        fig = plt.figure(1)
        ax1 = fig.subplots(1,1)
        ax1.set_title(f"Perceptron with {F.HType}inear data, noise={F.noise},pocket={F.pocket}")
        if F.HType != 'X':
            plt.ylim(-1, 1)
            plt.xlim(-1, 1)
        if F.HType == 'NL' or F.HType == 'X':
            angles = np.arange(0, 2*np.pi, 0.01)
            r0, rx, ry = weight_2_circle(F.w, angles)
            posIds = [ i for i in idAll if data[i,0]**2 + data[i,1]**2 >= r0**2 ]
            negIds = [ i for i in idAll if data[i,0]**2 + data[i,1]**2 < r0**2 ]
            ErrorList = [ i for i in idAll if np.sign(data[i,2]) != np.sign(yhat[i,0]) ]
##            ErrorListr = [ i for i in idAll if np.sign(data[i,2]) != np.sign(yhatr[i,0]) ]
            x1 = data[posIds,0]
            y1 = data[posIds,1]
            ax1.scatter(x1, y1, marker='+', c='green')
            x2 = data[negIds,0]
            y2 = data[negIds,1]
            ax1.scatter(x2, y2, marker='^', c='orange')
            x3 = data[ErrorList, 0]
            y3 = data[ErrorList, 1]
            ax1.scatter(x3, y3, marker='o', c='red')
            ax1.plot(rx, ry, label=f'F(x) circle radius {r0}',c='cyan',ls='solid',lw=2)
            ax1.legend()
            plt.show(block=False)
        elif F.HType == 'W':
            print("special W transform (no plot of ground truth)")
            print("regressed F.w:",F.w)
            yhat_reg = Wtilde_predict(F, data,plotting=True).T
            testa_w = np.array((-1,-0.05,0.08,0.13, 1.5, 1.5))
            testb_w = np.array((-1,-0.05,0.08,0.13, 1.5,  15))
            testc_w = np.array((-1,-0.05,0.08,0.13,  15, 1.5))
            testd_w = np.array((-1, -1.5,0.08,0.13,0.05,0.05))
            teste_w = np.array((-1,-0.05,0.08, 1.5,0.15,0.15))
            testf_w = np.array(( 1,    1,   1,   1,   1,   1))
            test_w = testc_w
            print("testing weight:",test_w)
            F.w = test_w.copy()
            yhat_test = Wtilde_predict(F, data,plotting=True).T
            print("shape of y_hat reg:",np.shape(yhat_reg))
            print("shape of y_hat test:",np.shape(yhat_test))
            ## Points grouped by Error(red x) vs Correct Pos(green)/Neg(blue)
            posIds = [ i for i in idAll if yhat_reg[i,0] >= 0 ]
            negIds = [ i for i in idAll if yhat_reg[i,0] < 0 ]
            ## TODO plot all three error lists and for all test cases
            ErrorList1 = [ i for i in idAll if np.sign(data[i,2]) != np.sign(yhat_reg[i,0]) ]
            ErrorList2 = [ i for i in idAll if np.sign(data[i,2]) != np.sign(yhat_test[i,0]) ]
            ErrorList = [ i for i in idAll if np.sign(yhat_reg[i,0]) != np.sign(yhat_test[i,0]) ]
            print("Total Errors:",len(ErrorList))
            if ErrorList != []:
                x1 = np.array(data[ErrorList, 0])
                y1 = np.array(data[ErrorList, 1])
                ax1.scatter(x1, y1, marker='o', c='red')
            if posIds != []:
                x2 = np.array(data[posIds, 0])
                y2 = np.array(data[posIds, 1])
                ax1.scatter(x2, y2, marker='+', c='green')
            if negIds != []:
                x3 = np.array(data[negIds, 0])
                y3 = np.array(data[negIds, 1])
                ax1.scatter(x3, y3, marker='x', c='orange')
        else:
            ## Points grouped by Error(red x) vs Polarity Pos(green)/Neg(blue)
            posIds = [ i for i in idAll if data[i,2] >= 0 ]
            negIds = [ i for i in idAll if data[i,2] < 0 ]
            ErrorList = [ i for i in idAll if np.sign(data[i,2]) != np.sign(yhat[i,0]) ]
            if ErrorList != []:
                x1 = np.array(data[ErrorList, 0])
                y1 = np.array(data[ErrorList, 1])
                ax1.scatter(x1, y1, marker='o', c='red',label='mismatch')
            if posIds != []:
                x2 = np.array(data[posIds, 0])
                y2 = np.array(data[posIds, 1])
                ax1.scatter(x2, y2, marker='+', c='green',label='positive')
            if negIds != []:
                x3 = np.array(data[negIds, 0])
                y3 = np.array(data[negIds, 1])
                ax1.scatter(x3, y3, marker='x', c='orange',label='negative')
            ax1.plot(xrange, yrange, label=f'F(x)={m1:2.2f}x+{b1:2.2f}',c='cyan',ls='solid',lw=2)
        return fig, ax1
    else:  ## TODO rewrite to generalize the transform
        if pla:
            if F.HType == 'L' or F.HType == 'X':
                ax1.plot(xrange, yrange, label=f'PLA: y={m1:2.2f}x+{b1:2.2f}',c='green',ls='dashed',lw=2)
            else:
                angles = np.arange(0, 2*np.pi, 0.01)
                r0, rx, ry = weight_2_circle(F.w, angles)
                ax1.plot(rx, ry, label=f'PLA: circle radius {r0}',c='green',ls='dashed',lw=2)
        elif lra:
            if F.HType == 'L':
                ax1.plot(xrange, yrange, label=f'LRA: y={m1:2.2f}x+{b1:2.2f}',c='orange',ls='dotted',lw=2)
            elif F.HType == 'X':
                F.w = tempH.copy()
                mtest, btest, yrange_test  = weight_2_line(F.w, xrange,bias=F.bias)
                ## TODO: Figure out whether to use LRA calc or not?
                ax1.plot(xrange, yrange_test, label=f'LRA: y={mtest:2.2f}x+{btest:2.2f}',c='orange',ls='dotted',lw=2)
            elif F.HType == 'NL':
                angles = np.arange(0, 2*np.pi, 0.01)
                r0, rx, ry = weight_2_circle(F.w, angles)
                ax1.plot(rx, ry, label=f'LRA: circle radius {r0}',c='orange',ls='dotted',lw=2)
        ax1.legend()
        plt.show(block=False)
    return

def plot_E1_vs_Eout(Ein, Eout, text):
    fig = plt.figure()
    ax1 = fig.subplots(1, sharex=True, sharey=True)
    ax1.set_title(text)
    N = len(Ein)
    M = len(Eout)
    x = np.zeros((N))
    y = np.zeros((N))
    z = np.zeros((M))
    Ein_c = np.cumsum(Ein)
    Eout_c = np.cumsum(Eout)
    for i in range(N):
        x[i] = i
        y[i] = Ein_c[i]/(i+1)
    for i in range(M):
        z[i] = Eout_c[i]/(i+1)
    ax1.plot(x, y, label=f"Ein mean = {np.mean(Ein)}")
    ax1.plot(x, z, label=f"Eout mean = {np.mean(Eout)}")
    ax1.legend()
    plt.show(block=False)


# =============================================================================
#%% File or Console I/O
"""
write_data
    I want this function to print:
    1. the 2-d data X (x1, x2),
    2. the observed value of y, (not just the sign), including noise
    3. h(x) without noise

write_params
    This function is for printing out params, hyperparams, and results 

analysis
    This function is for printing average scores both Ein/Eout, and their
    variances and any other results at the end of the main loop 

helper functions for writing out hyperparams and results
to either I/O or file 

"""
def write_data(outfile, L, Z, extra=False):
    for i in range(L.shape[0]):
        if extra:
            outputLine = f"{i}: F({L[i,0]},{L[i,1]}) = {Z[0,i]} vs noisy {L[i,2]}\n"
        else:            
            outputLine = f"{L[i,0]},{L[i,1]},{L[i,2]}\n"
        outfile.write(outputLine)
        
def writeout_learn_data(Target, M, record_of_w,record_of_x,record_of_h, record_of_correct, record_score):
    outfile = open("debug_learn.txt", 'w', newline='')
    outfile.write(f"Record of a learn: Target {Target.w}\n")
    outfile.write(f"Record of a learn: weights, x, H.h(x)\n")
    for i in range(M):
        outfile.write(f"{i} {record_of_w[i]}, {record_of_x[i]}, {record_of_h[i]} {record_of_correct[i]} {record_score[i]}\n")
    outfile.close()

        
def write_params(Param_f, hparams, results):
    # N    = rows of data per run
    file = open(Param_f, 'w', newline='')
    file.write(f"First Come the Hyperparams\n")
    # Write hyperparameters first (like HType)
    for i in range(len(hparams)):
        file.write(str(hparams[i]))
        file.write('\n')
    file.write(f"Next Come the Results\n")
    # results include E_in/E_out scores, etc...
    for j in range(len(results)):
        if j < 3:
            for k in range(len(results[j])):
                file.write(f"results[{j},{k}]={results[j][k]}\n" )
        else:
            file.write(f"results[{j}]={results[j]}\n")
    file.write(f"Finished.\n")
    file.close()
