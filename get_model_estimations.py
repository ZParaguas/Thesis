# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:30:16 2021

@author: PARAGUAS
"""

import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import t

from link_functions import *
from filter_functions import *
from llikelihood_functions import *

from display_estimation_results import *

np.set_printoptions(precision = 5)
plt.rcParams["figure.figsize"] = (16,9)

# =============================================================================
# Minimize LogLikelihoods
# =============================================================================


def fObjectiveUnivariateGarch(vTheta, vData):
    # initialize the parameter values

    dMu, dOmega, dAlpha, dBeta, dNu = fParameterTransformUnivariateGARCHt(vTheta)
    vTheta = [dMu, dOmega, dAlpha, dBeta, dNu]   
    vSig = fFilterUnivariateGarch(vTheta, vData)
    
    return fStudentTLikelihoodUnivariate(vData, dMu, vSig, dNu) 

def fObjectiveUnivariateGJRGarch(vTheta, vData):
    # initialize the parameter values
    dMu, dOmega, dAlpha, dBeta, dGamma, dNu = fParameterTransformUnivariateGJRGARCHt(vTheta)
    vTheta = [dMu, dOmega, dAlpha, dBeta, dGamma, dNu]   
    vSig = fFilterUnivariateGjrGarch(vTheta, vData)
    
    return fStudentTLikelihoodUnivariate(vData, dMu, vSig, dNu) 

def fObjectiveUnivariateGas(vTheta, vData):
    # initialize the parameter values
    dMu, dOmega, dAlpha, dBeta, dNu = fParameterTransformUnivariateGAS(vTheta)
    vTheta = [dMu, dOmega, dAlpha, dBeta, dNu]   
    vSig = fFilterUnivariateGas(vTheta, vData)
    
    return fStudentTLikelihoodUnivariate(vData, dMu, vSig, dNu) 

def fObjectiveStudentDccCopula(vTheta, mU, mEps):
    # initialize the parameter values

    dAlpha, dBeta, dNu = fParameterTransformCopulaDCC(vTheta)
    vTheta = [dAlpha, dBeta]
    
    mR = fFilterDccCopula(vTheta, mEps)
    
    return fStudentTCopulaLikelihood(mU, mR, dNu)

def fObjectiveStudentAdccCopula(vTheta, mU, mEps):
        
    dAlpha, dBeta, dGamma, dNu = fParameterTransformCopulaADCC(vTheta)
    vTheta = [dAlpha, dBeta, dGamma]
    
    mR = fFilterAdccCopula(vTheta, mEps)
    
    return fStudentTCopulaLikelihood(mU, mR, dNu)



# =============================================================================
# Estimate Marginals
# =============================================================================

def fEstimateMarginalsGarch(dfReturns, vTheta_star, options):
            
    vReturns1 = dfReturns[0]
    vReturns2 = dfReturns[1]
    
    dfResults1 = minimize(fObjectiveUnivariateGarch, vTheta_star, args=(vReturns1), options = options, method='BFGS')
    dfResults2 = minimize(fObjectiveUnivariateGarch, vTheta_star, args=(vReturns2), options = options, method='BFGS')

    return dfResults1, dfResults2

def fEstimateMarginalsGas(dfReturns, vTheta_star, options):
    
    vReturns1 = dfReturns[0]
    vReturns2 = dfReturns[1]
    
    dfResults1 = minimize(fObjectiveUnivariateGas, vTheta_star, args=(vReturns1), options = options, method='BFGS')
    dfResults2 = minimize(fObjectiveUnivariateGas, vTheta_star, args=(vReturns2), options = options, method='BFGS')

    return dfResults1, dfResults2

def fEstimateMarginalsGjrGarch(dfReturns, vTheta_star, options):
    
    vReturns1 = dfReturns[0]
    vReturns2 = dfReturns[1]
    
    dfResults1 = minimize(fObjectiveUnivariateGJRGarch, vTheta_star, args=(vReturns1), options = options, method='BFGS')
    dfResults2 = minimize(fObjectiveUnivariateGJRGarch, vTheta_star, args=(vReturns2), options = options, method='BFGS')
        
    return dfResults1, dfResults2
    
def fEstimateMarginals(dfReturns, vTheta_star_garch, vTheta_star_gjr_garch, vTheta_star_gas, options):
    
    dfResultsGarch1, dfResultsGarch2 = fEstimateMarginalsGarch(dfReturns, vTheta_star_garch, options)
    dfResultsGjrGarch1, dfResultsGjrGarch2 = fEstimateMarginalsGjrGarch(dfReturns, vTheta_star_gjr_garch, options)
    dfResultsGas1, dfResultsGas2 = fEstimateMarginalsGas(dfReturns, vTheta_star_gas, options)
        
    return {'GARCH':[dfResultsGarch1, dfResultsGarch2], 
            'GJR-GARCH': [dfResultsGjrGarch1, dfResultsGjrGarch2], 
            'GAS':[dfResultsGas1, dfResultsGas2]}

# =============================================================================
#  Copula's 
# =============================================================================

def fEstimateDccCopulas(mReturns, mSig_hat, vTheta_hat_marginals, dfTheta_star_dcc, options):
    
    vTheta1 = vTheta_hat_marginals[0]
    vTheta2 = vTheta_hat_marginals[1]
    
    dMu1 = vTheta1[0]
    dMu2 = vTheta2[0]

    dNu1 = vTheta1[-1]
    dNu2 = vTheta2[-1]
    
    # Initialize 
    vEps1 = (mReturns[0] - dMu1)/np.sqrt(mSig_hat[0]) / np.sqrt((dNu1 - 2)/dNu1)
    vEps2 = (mReturns[1] - dMu2)/np.sqrt(mSig_hat[1]) / np.sqrt((dNu2 - 2)/dNu2)
    mEps = np.stack((vEps1, vEps2)) 
    
    vU1 = t.cdf(vEps1, dNu1)
    vU2 = t.cdf(vEps2, dNu2)
    mU = np.stack((vU1, vU2)) 
    
    dfRes_student = minimize(fObjectiveStudentDccCopula, dfTheta_star_dcc, args=(mU, mEps), options = options, method='BFGS')
        
    return dfRes_student

def fEstimateAdccCopulas(mReturns, mSig_hat, vTheta_hat_marginals, dfTheta_star_adcc, options):
    
    vTheta1 = vTheta_hat_marginals[0]
    vTheta2 = vTheta_hat_marginals[1]
    
    dMu1 = vTheta1[0]
    dMu2 = vTheta2[0]

    dNu1 = vTheta1[-1]
    dNu2 = vTheta2[-1]
    
    # Initialize 
    vEps1 = (mReturns[0] - dMu1)/np.sqrt(mSig_hat[0]) / np.sqrt((dNu1 - 2)/dNu1)
    vEps2 = (mReturns[1] - dMu2)/np.sqrt(mSig_hat[1]) / np.sqrt((dNu2 - 2)/dNu2)
    mEps = np.stack((vEps1, vEps2)) 
    
    vU1 = t.cdf(vEps1 , dNu1)
    vU2 = t.cdf(vEps2 , dNu2)
    mU = np.stack((vU1, vU2)) 
                
    dfRes_student = minimize(fObjectiveStudentAdccCopula, dfTheta_star_adcc, args=(mU, mEps), options = options, method='BFGS')
        
    return dfRes_student


def fCopulaEstimation(mIn_sample_returns, dfUnivariate_predictions_train, dfTheta_hat, vTheta_star_dcc, vTheta_star_adcc, options):
   
    vTheta_hat_gjr_garch = dfTheta_hat['GJR-GARCH']
    vTheta_hat_gas = dfTheta_hat['GAS']

    dfResults_dcc_gjr = fEstimateDccCopulas(mIn_sample_returns, dfUnivariate_predictions_train['GJR-GARCH'], vTheta_hat_gjr_garch, vTheta_star_dcc, options)
    dfResults_dcc_gas = fEstimateDccCopulas(mIn_sample_returns, dfUnivariate_predictions_train['GAS'], vTheta_hat_gas, vTheta_star_dcc, options)
    
    dfResults_adcc_gjr = fEstimateAdccCopulas(mIn_sample_returns, dfUnivariate_predictions_train['GJR-GARCH'], vTheta_hat_gjr_garch, vTheta_star_adcc, options)
    dfResults_adcc_gas = fEstimateAdccCopulas(mIn_sample_returns, dfUnivariate_predictions_train['GAS'], vTheta_hat_gas, vTheta_star_adcc, options)
    
    return {'GJR-DCC':dfResults_dcc_gjr, 
            'GAS-DCC':dfResults_dcc_gas, 
            'GJR-ADCC':dfResults_adcc_gjr, 
            'GAS-ADCC':dfResults_adcc_gas}





