# Load necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import floor, log, exp, ceil, sqrt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from gekko import GEKKO

# Big issue with vectorization here
# Generalized logistic growht model, eqn (2)
# Input initial guess in form of an array K, r and A respectively, array name b
# time as variable t
# Output predicted number of cases y
# First value is the independent variable followed by the parameters
def log_growth(t,a,b,c):
    y = a/ (1 + c * exp(- (b * t)))
    return y

# Logistic growth rate, eqn (1)
# Input initial guess array, b
# Output rate of change of number of cases, dy
def log_growth_rate(t,a,b,c):
    dy1 = c * b * a * exp(-b*t)
    dy2 = ((1 + c * exp(-b*t)))**2
    dy = dy1/dy2 
    return dy

# calculate initial K, r, A using data from three equidistant points
# Input C -- data
# Output bo -- inital guess or empty list [] if calculation fails
def iniGuess(C):
    # k1, k2 and k3 are the tk, tk-m and tk - 2m from the paper
    k1 = 0
    k2 = 0
    k3 = 0
    b0 = [0,0,0] # Assign an empty array outputs to hold
    n = len(C)
    print(f"Total samples --> {n}")
    nmax = max(1,ceil(0.5 * n))
    print(f"Number of elements to consider for 3 equidistant point --> {nmax}")
    
    # calculate time interval for equidistant points: k-2*m, k-m, k
    # Here m is the interval size
    # In MATLAB, index starts from 1, but in python its 0
    # Hence we need to subtract 1 from len(C) to match python's index

    nindex = n -1

    # Dr Batista's equidistant point schema makes no sense
    # According to the paper, we take the first, middle and the last datapoint
    # In this software, I am hardcoding this value
    k1 = 0
    k3 = nindex
    k2 = int((k1 + k3)/2)
    m = k2 - k1 - 1     
    
    print(f"k1 -- {k1}, k2 -- {k2}, k3 -- {k3}, m -- {m}")
    print("Number of cases at chosen three points")
    print(f"k1 -- {C[k1]}, k2 -- {C[k2]}, k3 -- {C[k3]}")

    # Calculate numenator of eqn 11
    p = (C[k1] * C[k2]) - 2 * C[k1] * C[k3] + C[k2]*C[k3]
    if (p<=0):
        p = 0
    
    # Calculate denomenator of eqn 11
    q = (C[k2]*C[k2]) - C[k3]*C[k1]
    if (q<=0):
        q = 0
    
    # Population number cannot be float
    K = int(C[k2] * (p/q)) 
    if (K<0):
        K = max(C)
    
    # Calculate r using eqn 12
    r1 = C[k3] * (C[k2] - C[k1])
    r2 = C[k1] * (C[k3] - C[k2])
    r = (1/m) * log(r1/r2)
    if (r<0 or r == float("inf")):
        r = 0.5
    
    # Calculate r using eqn 13
    A1 = ((C[k3] - C[k2])*(C[k2] - C[k1])) / ((C[k2]*C[k2])-C[k3]*C[k1])
    A2 = (C[k3] * (C[k2] - C[k1])) / (C[k1] * (C[k3] - C[k2]))
    #print (A1, A2)
    #A2 = A2 ** ((k3 - m)/m)
    A22 = (k3 - m)/m
    A2 = pow(A2, A22)
    A = A1 * A2
    if (A < 0 or A == float("inf")):
        A = max(C)
    # Max capacity of a population cannot be a float number
    A = int(A)

    # Print debug report
    print(f"p -- {p}")
    print(f"q -- {q}")
    print(f"r -- {r}")
    print(f"A -- {A}")

    print(f"Initial K value -- {K}")
    print(f"Initial r value -- {r}")
    print(f"Initial A value -- {A}")

    # Load initial guesses
    b0[0] = K
    b0[1] = r
    b0[2] = A
    
    return b0


# Nonlinear least squares fit function
def gekko_nonlinearfit(timestamp, sampleC, c0):
    m=GEKKO(remote=False)
    xm = timestamp
    ym = sampleC
    x = m.Param(value=xm)
    y = m.CV(value = ym)
    y.FSTATUS = 1
    a = m.FV(value = c0[0])
    a.STATUS = 1
    b = m.FV(value = c0[1])
    b.STATUS = 1
    c = m.FV(value = c0[2])
    c.STATUS = 1

    m.Equation(y == a/(1 + c * m.exp(-b*x)))

    m.options.EV_TYPE = 2 # L2 norm, sum of squared error
    #m.options.MAX_ITER = 100 # Sets the number of iteration
    m.options.IMODE = 2 # Bascially an equivalent to Least-square optimization
    m.solve(disp = False)

    #print(a.value[0],b.value[0], c.value[0])
    print(f"Regression parameters are K = {a.value[0]}, r = {b.value[0]}, A = {c.value[0]}")
    return a.value[0], b.value[0], c.value[0]

## Plot rate of change curve
def plot_rate_change(params, sampleC, timestamp, plot_on):
    yychange = []
    for j in timestamp:
        yy2 = log_growth_rate(j, params[0], params[1], params[2])
        yychange.append(yy2)
    
    if (plot_on == True):
        fig = plt.figure(figsize=(10,5))
        ddC = abs(np.diff(sampleC)) # Find difference between succesive days 
        #print(timestamp[:(len(sampleC)-1)])
        plt.bar(timestamp[:(len(sampleC)-1)],ddC, label = 'Difference in cumulative cases') # Array slicing
        plt.plot(timestamp, yychange, 'r-', label = 'Logistic Growth Rate')
        plt.legend()
        plt.show()

    return yychange

## Plot data vs predicted number of cases, no phases
def plot_cumulative_fit(params, sampleC, timestamp, plot_on):
    yypred = []
    for i in timestamp:
        yyy = log_growth(i, params[0], params[1], params[2])
        yypred.append(yyy)
    if (plot_on == True):
        fig = plt.figure(figsize=(10,5))
        plt.plot(timestamp, sampleC, 'mo', label = 'Data')
        plt.plot(timestamp, yypred, 'r-', label = 'Logistic Growth Model')
        plt.legend()
        plt.show()
    
    return yypred

## Plot figure with phases described by Dr Batista
def plot_cumu_phases(params, timestamp, sampleC, yypred,I_change):
    Kopt = params[0]
    ropt = params[1]
    Aopt = params[2]

    neg_lim = -5 # To make sure we can see the yaxis scale clearly
    hhh = max(sampleC) + 10

    fig = plt.figure(figsize=(15,7))
    
    # Fix x and y limits
    #plt.xlim(0,60)
    #plt.ylim(0,1)

    #plt.xlim(neg_lim,(len(timestamp)+1))
    #plt.ylim(neg_lim,hhh, max(sampleC) + 20)
    
    ax = fig.add_subplot(1,1,1) # one row, one column, first plot

    # Perform min-max normalization
    y_act_scaled = min_max_norm(sampleC)
    y_pred_scaled = min_max_norm(yypred)
    I_pred_scaled = min_max_norm(I_change)
    print(timestamp.shape)

    # Draw data points, logistic growth curve fit and logistic growth rate
    #plt.scatter(timestamp, y_act_scaled, 'ko', label = 'Data')
    ax.plot(timestamp, y_act_scaled, 'ko', label = 'Data')
    ax.plot(timestamp, y_pred_scaled, 'b-', label = 'Cumulative fit',lw = 5)
    ax.plot(timestamp, I_pred_scaled, 'r-', label = 'Rate of change fit',lw = 5)

    tpeak = int(log(Aopt)/ropt)
    #print(tpeak)

    tp2 = int(tpeak - (2/ropt))
    tp3 = int(tpeak + (2/ropt))
    tp4 = int(tpeak + 2 * (2/ropt))

    phase1_x = abs(0 - tp2)
    phase2_x = abs(tp2 - tpeak)
    phase3_x = abs(tpeak - tp3)
    phase4_x = abs(tp3 - tp4)
    phase5_x = abs(tp4 - len(timestamp))

    hhh_adjust = hhh + (-1 * neg_lim) # To make sure all the rectangles are of the same height

    # HARDCODED, change hex values to change color
    # RGB color values for each rectangles
    # Hex values found using google's rgb to hex converter
    # Type "rgb to hex" in a google search bar to bring up the pallet

    phase1_color = '#d8d4d9'
    phase2_color = '#ed8a58'
    phase3_color = '#e173eb'
    phase4_color = '#f7f494'
    phase5_color = '#b4f781'

    # Create rectagle objects
    phase_rect1 = patches.Rectangle((0,neg_lim), phase1_x, hhh_adjust, color=phase1_color)
    phase_rect2 = patches.Rectangle((tp2,neg_lim), phase2_x, hhh_adjust, color=phase2_color)
    phase_rect3 = patches.Rectangle((tpeak,neg_lim), phase3_x, hhh_adjust, color=phase3_color)
    phase_rect4 = patches.Rectangle((tp3,neg_lim), phase4_x, hhh_adjust, color=phase4_color)
    phase_rect5 = patches.Rectangle((tp4,neg_lim), phase5_x, hhh_adjust, color=phase5_color)

    # Add rectangle objects
    ax.add_patch(phase_rect1)
    ax.add_patch(phase_rect2)
    ax.add_patch(phase_rect3)
    ax.add_patch(phase_rect4)
    ax.add_patch(phase_rect5)


    # Add textbox to name each rectangle
    color_list = [phase1_color,phase2_color,phase3_color,phase4_color,phase5_color] # List to hold each rectangle's color

    xxx_text_box = [(phase1_x/2),(tpeak - phase2_x/2),(tpeak + phase3_x/2),(tp4 - phase4_x/2),(tp4 + phase5_x/2)] # X coordinates for each textbox

    phase_str_list = ['1','2','3','4','5'] # Strings to show in the textbox

    # HARDCODED
    hhh_text_box = max(y_act_scaled) + 0.1

    # Loop to print textboxes
    for kk in range(5):
        #textstr = 'Phase' + " " + str(kk) # Doesn't fit in the rectangles
        textstr = phase_str_list[kk]
        props = dict(boxstyle='round', facecolor='azure', alpha=0.5)
        ax.text(xxx_text_box[kk],hhh_text_box, textstr, fontsize=15, verticalalignment='top', bbox=props)
    
    # Print a red line which shows at which day, the peak growth rate occured 
    # HARDCODED, label
    plt.axvline(x = tpeak, lw = 2, color = 'red', label = 'Max growth rate at week {} '.format(tpeak))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=4)

    # Save figure
    plt.savefig('combined_phase.png')
    plt.show()

# Reference - https://learn.64bitdragon.com/articles/computer-science/data-processing/min-max-normalization
def min_max_norm(x):
    min_num = np.min(x)
    max_num = np.max(x)
    range_num = max_num - min_num
    aa = [((a - min_num) / range_num) for a in x]
    aa = np.asarray(aa)
    return aa

## --------------------- Unused code --------------- ##

