# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 17:07:02 2021

@author: PARAGUAS
"""
import matplotlib.pyplot as plt
import numpy as np
import math  
from scipy.stats import t
from scipy.linalg import sqrtm
from link_functions import *


def fCalculateResiduals(mReturns, mSig_hat, mTheta_hat):
    
    vTheta_hat1 = mTheta_hat[0]
    vTheta_hat2 = mTheta_hat[1]
    
    dMu1 = vTheta_hat1[0]
    dMu2 = vTheta_hat2[0]

    dNu1 = vTheta_hat1[-1]
    dNu2 = vTheta_hat2[-1]
    
    # Initialize 
    vEps1 = (mReturns[0] - dMu1)/np.sqrt(mSig_hat[0]) / np.sqrt((dNu1 - 2)/dNu1)
    vEps2 = (mReturns[1] - dMu2)/np.sqrt(mSig_hat[1]) / np.sqrt((dNu2 - 2)/dNu2)
    mEps = np.stack((vEps1, vEps2)) 
    
    
    return mEps

def fCalculatePIT(mEps, mTheta_hat):
    
    vTheta_hat1 = mTheta_hat[0]
    vTheta_hat2 = mTheta_hat[1]

    dNu1 = vTheta_hat1[-1]
    dNu2 = vTheta_hat2[-1]
    
    # Initialize 
    vU1 = t.cdf(mEps[0], dNu1)
    vU2 = t.cdf(mEps[1], dNu2)
    
    mU = np.stack((vU1, vU2)) 
    
    return mU



# =============================================================================
# Score functions
# =============================================================================

def fStudentsTScoreUnivariate(dF, dY, dMu, dNu):
    
    scaled_score = -0.5 + ((dNu + 1)/2)*((dY - dMu)**2)/(((dNu - 2)*dF) + (dY - dMu)**2)
       
    #scaled_score = (dNu + 3)/dNu * ( (1 + ((dY-dMu)**2)/(dNu - 2))**-1 * (((dNu + 1)/(dNu - 1)) * ((dY - dMu)**2) / (dF) - 1))
    
    return scaled_score

# =============================================================================
# 
# =============================================================================


def fFilterUnivariateGarch(vTheta, vData):
    # initialize the parameter values
    dMu, dOmega, dAlpha, dBeta, dNu = vTheta[0], vTheta[1], vTheta[2], vTheta[3], vTheta[4] 

    T = len(vData)
    sig1 = np.var(vData) # initial value for conditional volatility
    eps1 = np.mean(vData) # initial value for conditional volatility
    
    
    vSig = np.zeros(T)
    vEpsilon = np.zeros(T)
    
    ## 5. Define Initialization for Time Series
    vSig[0] = sig1
    vEpsilon[0] = eps1

    ## 6. Generate Time Series
    
    for j in range(0, T-1):
        vSig[(j+1)] = dOmega + dAlpha * vEpsilon[j]**2 + dBeta * vSig[j]
        vEpsilon[(j+1)] = vData[(j+1)] - dMu - dPhi * vData[j]
        
    return vSig 


def fFilterUnivariateGjrGarch(vTheta, vData):
    # initialize the parameter values
    dMu, dOmega, dAlpha, dBeta, dGamma, dNu = vTheta[0], vTheta[1], vTheta[2], vTheta[3], vTheta[4], vTheta[5]

    T = len(vData)

    vSig = np.zeros(T)
    ## 5. Define Initialization for Time Series
    vSig[0] = (np.var(vData))

    ## 6. Generate Time Series
    
    for j in range(0, T-1):
        if vData[j]-dMu <= 0:
            iInd = 1
        else:
            iInd = 0

        vSig[(j+1)] = dOmega + dAlpha * (vData[j] - dMu)**2 + dBeta * vSig[j] + dGamma * iInd * (vData[j] - dMu)**2

    return (vSig)

def fFilterUnivariateGas(vTheta, vData):
    # initialize the parameter values
    dMu, dOmega, dAlpha, dBeta, dNu = vTheta[0], vTheta[1], vTheta[2], vTheta[3], vTheta[4] 

    iT = len(vData)
            
    vF = np.zeros(iT)
    score = np.zeros(iT)
    
    vF[0] = np.log(np.var(vData))
    
    for i in range(iT-1):

        score[i] = fStudentsTScoreUnivariate(np.exp(vF[i]), vData[i], dMu, dNu)
        vF[i+1] = dOmega + dAlpha * score[i] + dBeta * vF[i]
        
    return np.exp(vF)


# =============================================================================
# Updating equations copulas
# =============================================================================
    
def fFilterDccCopula(vTheta, mEps):
    # initialize the parameter values
    
    dAlpha, dBeta = vTheta[0], vTheta[1]

    T = len(mEps[0])

    mQ = np.zeros((T, 2, 2))
    mR = np.zeros((T, 2, 2))
    
    mQ_bar = np.corrcoef(mEps[0], mEps[1])    
    mQ[0] = mQ_bar
    mR[0] = mQ_bar
        
    for j in range(0, T-1):
        mEps_t = (mEps[:,j] * mEps[:,j].reshape((2, 1)))
      
        mQ[j+1] = mQ_bar * (1 - dAlpha - dBeta) + dAlpha * mEps_t + dBeta * mQ[j]   

        vRho = mQ[(j+1), 1, 0]/np.sqrt(mQ[(j+1), 0, 0] * mQ[(j+1), 1, 1])
        mR[j+1] = np.array([[1, vRho], [vRho, 1]]) 
        
    return mR

def fFilterAdccCopula(vTheta, mEps):
    # initialize the parameter values

    dAlpha, dBeta, dGamma = vTheta[0], vTheta[1], vTheta[2]

    T = len(mEps[0])

    mEta = mEps
    mEta[mEta >= 0] = 0

    mQ = np.zeros((T, 2, 2))
    mR = np.zeros((T, 2, 2))
        
    mQ_bar = np.corrcoef(mEps[0], mEps[1]) 
    mN_bar = np.corrcoef(mEta[0], mEta[1])
        
    mQ[0] = mQ_bar
    mR[0] = mQ_bar
            
    for j in range(0, T-1):
        mEps_t = (mEps[:,j] * mEps[:,j].reshape((2, 1)))
        mEta_t = (mEta[:,j] * mEta[:,j].reshape((2, 1)))

        mQ[j+1] = mQ_bar*(1 - dAlpha - dBeta) + dAlpha * mEps_t + dBeta * mQ[j] + dGamma*(mEta_t - mN_bar)

        vRho = mQ[(j+1), 1, 0]/np.sqrt(mQ[(j+1), 0, 0] * mQ[(j+1), 1, 1])
        mR[j+1] = np.array([[1, vRho], [vRho, 1]]) 
        
    return mR





def fGetUnivariatePredictions(vTickers, mIn_sample_returns, dfTheta_hat):
        
    vTheta_garch = dfTheta_hat['GARCH']
    vTheta_gjr_garch = dfTheta_hat['GJR-GARCH']
    vTheta_gas = dfTheta_hat['GAS']
    
    dfReturns_1 = mIn_sample_returns[0]
    dfReturns_2 = mIn_sample_returns[1]
    
    vSig_garch1 = fFilterUnivariateGarch(vTheta_garch[0], dfReturns_1)
    vSig_garch2 = fFilterUnivariateGarch(vTheta_garch[1], dfReturns_2)

    vSig_gjr1 = fFilterUnivariateGjrGarch(vTheta_gjr_garch[0], dfReturns_1)
    vSig_gjr2 = fFilterUnivariateGjrGarch(vTheta_gjr_garch[1], dfReturns_2)

    vSig_gas1 = fFilterUnivariateGas(vTheta_gas[0], dfReturns_1)
    vSig_gas2 = fFilterUnivariateGas(vTheta_gas[1], dfReturns_2)

    return {'GARCH':[vSig_garch1, vSig_garch2], 
                'GJR-GARCH':[vSig_gjr1, vSig_gjr2],
                'GAS':[vSig_gas1, vSig_gas2]}

def fGetDynamicCorrelations(mReturns, dfUnivariate_predictions_test, dfTheta_hat_marginals, dfTheta_hat_copulas):
        
    dfTheta_gjr = dfTheta_hat_marginals['GJR-GARCH']
    dfTheta_gas = dfTheta_hat_marginals['GAS']
    
    vTheta_hat_gjr_dcc = dfTheta_hat_copulas['GJR-DCC']
    vTheta_hat_gas_dcc = dfTheta_hat_copulas['GAS-DCC']
    
    vTheta_hat_gjr_adcc = dfTheta_hat_copulas['GJR-ADCC']
    vTheta_hat_gas_adcc = dfTheta_hat_copulas['GAS-ADCC']

    mEps_gjr = fCalculateResiduals(mReturns, dfUnivariate_predictions_test['GJR-GARCH'], dfTheta_gjr)
    mEps_gas = fCalculateResiduals(mReturns, dfUnivariate_predictions_test['GAS'], dfTheta_gas)
    
    vRho_gjr_dcc =  fFilterDccCopula(vTheta_hat_gjr_dcc, mEps_gjr)               
    vRho_gas_dcc =  fFilterDccCopula(vTheta_hat_gas_dcc, mEps_gas)
               
    vRho_gjr_adcc = fFilterAdccCopula(vTheta_hat_gjr_adcc, mEps_gjr)
    vRho_gas_adcc = fFilterAdccCopula(vTheta_hat_gas_adcc, mEps_gas)


    return {'GJR-DCC':vRho_gjr_dcc, 
            'GJR-ADCC':vRho_gjr_adcc, 
            'GAS-DCC': vRho_gas_dcc,
            'GAS-ADCC': vRho_gas_adcc,
            }


