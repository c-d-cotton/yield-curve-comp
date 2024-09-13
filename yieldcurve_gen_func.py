#!/usr/bin/env python3
import datetime
import functools
import math
import numpy as np
from numpy import exp
import pandas as pd
from scipy.optimize import brentq
import statsmodels.api as sm
import matplotlib.pyplot as plt
from nelson_siegel_svensson.calibrate import calibrate_ns_ols

# General Yield Functions:{{{1
def getyieldfromprice_aux(price, coupon, years, grossinterestrate):
    """
    The interestrate is such that this equation = 0
    """
    couponpayments = np.sum([0.5 * coupon / grossinterestrate ** (halfyeari / 2) for halfyeari in range(years * 2)])
    price2 = couponpayments + 1 / grossinterestrate ** years
    return(price - price2)

    
def getyieldfromprice(price, coupon, years):
    """
    Input the price for a bond with face value of $1 and a biannual coupon of coupon/2 with a maturity in years

    Note: coupons are paid in the last period of the bond's life: https://corporatefinanceinstitute.com/resources/knowledge/trading-investing/coupon-bond/

    stuff to check:
    - coupon done contnuously? i.e. what if bond 3/4 of year long
    """
    f1 = functools.partial(getyieldfromprice_aux, price, coupon, years)
    # returns the gross interest rate
    interestrate = brentq(f1, 0.5, 1.5)

    interestrate = (interestrate - 1)
    interestrate = interestrate * 100
    round(interestrate, 4)
    return(interestrate)


def getyieldfromprice_test():
    rate = getyieldfromprice(1, 0, 10)
    print(rate)
    rate = getyieldfromprice(1, 0.1, 1)
    print(rate)


# OLS Yield Curve Functions:{{{1
def getolsycparam(years, yields, dropabove = 10):
    """
    Yields in percentages
    Corresponding year maturities of bonds
    """
    # if dropabove is not None:
    #     keepi = [i for i in range(len(years)) if years[i] <= dropabove]
    #     yields = [yields[i] for i in keepi]
    #     years = [years[i] for i in keepi]

    y = np.array(yields)
    X = np.array(years)
    # print(X)
    X = sm.add_constant(X)

    params = sm.OLS(y, X).fit().params

    return(params)


def ols_yieldcurve(maturities_input, yields_input, maturities_output):
    """
    Get the estimated average interest rate for 0-1, 1-2, ..., 14-15 years
    """
    params = getolsycparam(maturities_input, yields_input)

    alpha = params[0]
    beta = params[1]

    yields_output = []
    for maturity in maturities_output:
        yields_output.append(alpha + beta * maturity)

    return(params, yields_output)


def ols_yieldcurve_test():
    maturitiesinyears_us = [.0833, .167, .25, .5, 1, 2, 3, 5, 7, 10, 20, 30]
    annualyields_us = [.0005, .0005, .0005, .0006, .0007, .0017, .0033, .0066, .0096, 0.012, 0.0177, 0.0186]

    params, yields_output = ols_yieldcurve(maturitiesinyears_us, annualyields_us, maturitiesinyears_us)
    print(yields_output)

# ols_yieldcurve_test()
# Ridge Yield Curve:{{{1
def ridgelist_yieldcurve(maturities_input, yields_input, maturities_output, yearwindow):
    """
    This function returns a yield curve where I take the yield of maturities directly if there's a similar maturity nearby
    Or I take the midpoint of two maturities before and after if they're not too far away
    Or I leave it blank if neither is possible

    yearwindow is the width I consider when forming the ridges. yearwindow = 1 means around a maturityoutput of 5, I'd include the closest bond to having a maturity of 5 within the window 4.5-5.5
    """
    # none of the yields_input should be nan otherwise the function won't work
    for i in range(len(yields_input)):
        if np.isnan(yields_input[i]):
            raise ValueError("Should not be nan in yields_input.")

    # output list
    yields_output = [np.nan] * len(maturities_output)
    for matouti, matout in enumerate(maturities_output):

        # first we look for a number that is close enough to matout to read the yield directly from
        low = matout - yearwindow / 2
        high = matout + yearwindow / 2

        thisyieldsoutput = []

        for matini, matin in enumerate(maturities_input):
            if matin >= low and matin <= high:
                thisyieldsoutput.append(yields_input[matini])

        if len(thisyieldsoutput) > 0:
            yields_output[matouti] = np.mean(thisyieldsoutput)
        

    return(yields_output)
    

def ridgelist_yieldcurve_test1():
    maturitiesinyears_us = [.0833, .167, .25, .5, 1, 2, 3, 5, 7, 10, 20, 30]
    annualyields_us = [.0003, .0004, .0005, .0006, .0007, .0017, .0033, .0066, .0096, 0.012, 0.0177, 0.0186]

    # now testing actual interpolation
    print('with window of 1 year')
    yields_output = ridgelist_yieldcurve(maturitiesinyears_us, annualyields_us, list(range(1, 16)), 1)
    print(yields_output)

    print('with window of 2 year')
    yields_output = ridgelist_yieldcurve(maturitiesinyears_us, annualyields_us, list(range(1, 16)), 2)
    print(yields_output)


def ridgelist_yieldcurve_test2():
    maturitiesinyears = [3, 7]
    annualyields = [0.01, 0.03]

    # now testing actual interpolation
    print('with window of 1 year')
    yields_output = ridgelist_yieldcurve(maturitiesinyears, annualyields, list(range(1, 16)), 1)
    print(yields_output)

    print('with window of 2 year')
    yields_output = ridgelist_yieldcurve(maturitiesinyears, annualyields, list(range(1, 16)), 2)
    print(yields_output)


# Ridge Yield Curve:{{{1
def ridge_yieldcurve(maturities_input, yields_input, matout_low, matout_high, matout_string = False):
    """
    Here I specify the windows I want to output
    So matout_low/matout_high should be something like 7.5-9.5
    Then I'll find the average of the yields_input with maturities in that range
    """
    # none of the yields_input should be nan otherwise the function won't work
    for i in range(len(yields_input)):
        if np.isnan(yields_input[i]):
            raise ValueError("Should not be nan in yields_input.")

    # adjust if matout_str is True
    if matout_string is True:
        matout_low_letter = matout_low[0]
        matout_low = int(matout_low[1: ])
        if matout_low_letter == 'y':
            None
        elif matout_low_letter == 'z':
            None
        elif matout_low_letter == 'm':
            matout_low = matout_low / 12
        elif matout_low_letter == 'd':
            matout_low = matout_low / 365
        else:
            raise ValueError("matout_low misspecified: " + matout_low_letter + ".")
        matout_high_letter = matout_high[0]
        matout_high = int(matout_high[1: ])
        if matout_high_letter == 'y':
            None
        elif matout_high_letter == 'z':
            None
        elif matout_high_letter == 'm':
            matout_high = matout_high / 12
        elif matout_high_letter == 'd':
            matout_high = matout_high / 365
        else:
            raise ValueError("matout_high misspecified: " + matout_high_letter + ".")

    if matout_low >= matout_high:
        raise ValueError("matout_low and matout_high misspecified.")

    thisyieldsoutput = []

    for matini, matin in enumerate(maturities_input):
        if matin >= matout_low and matin <= matout_high:
            thisyieldsoutput.append(yields_input[matini])

    if len(thisyieldsoutput) > 0:
        rateoutput = np.mean(thisyieldsoutput)
    else:
        rateoutput = np.nan
        
    return(rateoutput)


def ridge_yieldcurve_test():
    maturitiesinyears_us = [.0833, .167, .25, .5, 1, 2, 3, 5, 7, 10, 20, 30]
    annualyields_us = [.0003, .0004, .0005, .0006, .0007, .0017, .0033, .0066, .0096, 0.012, 0.0177, 0.0186]

    # now testing actual interpolation
    print('0-1years - should be 0.0005')
    rateoutput = cridge_yieldcurve(maturitiesinyears_us, annualyields_us, 'm00', 'm12', matout_string = True)
    print(rateoutput)

    print('1-2.9 - should be 0.0012')
    rateoutput = cridge_yieldcurve(maturitiesinyears_us, annualyields_us, 1, 2)
    print(rateoutput)


# Nelson Siegel My Version:{{{1
def explinspace(low, high, numval):
    """
    function to define exponential grid
    """
    if low <= 0:
        raise ValueError('low should be >0.')
    logvals = np.linspace(np.log(low), np.log(high), numval)
    vals = np.exp(logvals)

    return(vals)

def nsme_singlereg(maturitiesinyears, annualyields, tau, prnt_results = False):
    """
    performs a single regression, returns the regression
    prnt_results: option to print the regression summary
    """

    df = pd.DataFrame({'maturity': maturitiesinyears, 'yield': annualyields})

    df['constant'] = 1
    df['x1'] = (1 - np.exp(-df['maturity'] / tau)) / (df['maturity'] / tau )
    df['x2'] = np.exp( -df['maturity'] / tau )
    
    y = df['yield']
    X = df[['constant', 'x1', 'x2']]

    # adds constant and performs regression
    # x_vars_con = sm.add_constant(x_vars)
    nsme_model = sm.OLS(y, X)
    nsme_reg = nsme_model.fit()

    r2 = nsme_reg.rsquared

    params = nsme_reg.params
    beta0 = params['constant']
    beta2 = - params['x2']
    beta1 = params['x1'] - beta2

    return(r2, beta0, beta1, beta2, tau)


def nsme_singlereg_test():
    nsme_singlereg([1, 2, 3, 4, 5], [0.01, 0.02, 0.03, 0.041, 0.05], 100)


def nsme_calib_single(maturitiesinyears, annualyields, iterations = 20):
    """
    This calls the nsme_singlereg function over a wide range of tau and picks the best tau
    Returns the associated parameters
    """
    taus = explinspace(0.01, 100, iterations)

    results = None
    for tau in taus:
        iterationresults = nsme_singlereg(maturitiesinyears, annualyields, tau)

        # if R2 rises then use this iteration as the results
        if results is None or iterationresults[0] > results[0]:
            results = iterationresults

    return(results[1: ])

    
def nsme_calib_single_test():
    maturitiesinyears_us = [.0833, .167, .25, .5, 1, 2, 3, 5, 7, 10, 20, 30]
    annualyields_us = [.0005, .0005, .0005, .0006, .0007, .0017, .0033, .0066, .0096, 0.012, 0.0177, 0.0186]

    params = nsme_calib_single(maturitiesinyears_us, annualyields_us)
    print(params)

# nsme_calib_single_test()
def nsme_yield(maturity, params):
    """
    defines sn yield curve function
    """
    beta0, beta1, beta2, tau = params

    theyield = beta0 + (beta1 + beta2) * (1 - np.exp( -maturity/tau ) ) / (maturity / tau) - beta2 * np.exp(-maturity/tau)

    return(theyield)


def nsme_yieldcurve(maturities_input, yields_input, maturities_output):
    params = nsme_calib_single(maturities_input, yields_input)

    yields_output = [nsme_yield(maturity, params) for maturity in maturities_output]

    return(params, yields_output)
    

def nsme_yieldcurve_test():
    maturitiesinyears_us = [.0833, .167, .25, .5, 1, 2, 3, 5, 7, 10, 20, 30]
    annualyields_us = [.0005, .0005, .0005, .0006, .0007, .0017, .0033, .0066, .0096, 0.012, 0.0177, 0.0186]

    params, yields_output = nsme_yieldcurve(maturitiesinyears_us, annualyields_us, maturitiesinyears_us)
    print(yields_output)

# nsme_yieldcurve_test()
# Nelson Siegel - Luphord Version:{{{1
def nsluphord_test():
    """
    Testing the yield curve availalbe at https://github.com/luphord/nelson_siegel_svensson
    """
    from nelson_siegel_svensson.calibrate import calibrate_ns_ols

    t = np.array([0, 1, 2, 3])
    y = np.array([0, 1, 2, 3])
    curve, status = calibrate_ns_ols(t, y, tau0 = 1.0)
    print(curve)
    print(status)
    print(curve(np.array([0, 1, 2, 3])))


def nsluphord_yieldcurve(maturities_input, yields_input, maturities_output):
    curve, status = calibrate_ns_ols(np.array(maturities_input), np.array(yields_input), tau0 = 1.0)
    yields_output = curve(np.array(maturities_output))
    return(vars(curve), yields_output)


def nsluphord_yieldcurve_test():
    maturitiesinyears_us = [.0833, .167, .25, .5, 1, 2, 3, 5, 7, 10, 20, 30]
    annualyields_us = [.0005, .0005, .0005, .0006, .0007, .0017, .0033, .0066, .0096, 0.012, 0.0177, 0.0186]

    params, yields_output = nsluphord_yieldcurve(maturitiesinyears_us, annualyields_us, maturitiesinyears_us)

    print(yields_output)


# nsluphord_yieldcurve_test()
# Yield Curve - Comparison:{{{1
def yc_comparison():
    """
    Comparing different methods to construct yield curves.
    """

    print('Weird Case 1:')
    maturities = [1.0, 10.0, 20.0, 25.0, 30.0]
    yields = [0.0213, 0.0412, 0.0723, 0.0645, 0.0618]
    print('NS Me:')
    output = nsme_yieldcurve(maturities, yields, list(range(1, 20)))
    print(output[0])
    print(output[1])
    print('NS Luphord:')
    output = nsluphord_yieldcurve(maturities, yields, list(range(1, 20)))
    print(output[0])
    print(output[1])
    # nsluphord has negative yields between 1 and 10 whereas nsme does not
    # no negative yields probably makes more sense

    print('\nWeird Case 2:')
    maturities = [1.0, 3.0, 5.0, 9.0]
    yields = [0.03114, 0.042226000000000006, 0.02381, 0.024476]
    print('NS Me:')
    output = nsme_yieldcurve(maturities, yields, list(range(1, 20)))
    print(output[0])
    print(output[1])
    print('NS Luphord:')
    output = nsluphord_yieldcurve(maturities, yields, list(range(1, 20)))
    print(output[0])
    print(output[1])
    # these ones are more similar

    print('\nTime Taken:')
    maturities = [1.0, 10.0, 20.0, 25.0, 30.0]
    yields = [0.0213, 0.0412, 0.0723, 0.0645, 0.0618]
    start = datetime.datetime.now()
    nsme_yieldcurve(maturities, yields, list(range(1, 20)))[1]
    print('nsme took: ' + str(datetime.datetime.now() - start))
    start = datetime.datetime.now()
    nsluphord_yieldcurve(maturities, yields, list(range(1, 20)))[1]
    print('nsluphord took: ' + str(datetime.datetime.now() - start))
    start = datetime.datetime.now()
    ols_yieldcurve(maturities, yields, list(range(1, 20)))[1]
    print('ols took: ' + str(datetime.datetime.now() - start))
    start = datetime.datetime.now()
    ridge_yieldcurve(maturities, yields, list(range(1, 20)))[1]
    print('ridge took: ' + str(datetime.datetime.now() - start))

# yc_comparison()
