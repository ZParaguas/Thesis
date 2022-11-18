# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 16:08:13 2021

@author: PARAGUAS
"""


import numpy as np
import scipy as sc
import sys
import pandas as pd
import yfinance as yf

import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import t
from scipy.stats import norm

import seaborn as sns
sns.set()

def fStudentTLikelihoodUnivariate(vData, dMu, vSigma2, dNu):
    # Compute the likelihood function of a univariate Student's t distribution
    #
    # INPUTS:
    # vData: a 1 x n vector containing the data
    # dMu: a scalar containing the location parameter
    # dSigma2: a scalar containing the squared scale parameter
    # dNu: a scalar containing the degrees of freedom parameter
    #
    # OUTPUT:
    # llik: a scalar containing the log likelihood
    #
  
    iN = len(vData)
    llik = (
        iN * (
            sc.special.loggamma(0.5*(dNu + 1)) - 
            sc.special.loggamma(0.5*dNu) -
            0.5 * np.log(np.pi*(dNu - 2))
        ) -
            0.5 * np.sum(np.log(vSigma2))
         - (
            0.5 * (dNu + 1) * np.sum(np.log(1 + np.square(vData - dMu)/((dNu - 2) * vSigma2)))
        )
    )
    
    if np.isnan(llik):
        llik = -1e-6
    return -llik/iN


def fStudentTCopulaLikelihood(mU, mR, dNu):
    # Compute the Student's t(nu) copula likelihood function
    #
    # INPUTS:
    # mU: a p x n matrix containing the data, formulated as PITs
    # mR: a p x p correlation matrix
    # dNu: a sccalar containing the degrees of freedom parameter
    #
    # OUTPUT:
    # llik: a scalar containing the log copula likelihood
    #

    # initialize the dimensions of the data
    iP = mU.shape[0]
    iN = mU.shape[1]
    
    # transform the PITs to pseudo Student t data
    mDataTilde = t.ppf(mU, dNu)
     
    llik_total = 0
    for i in range(1, iN):
        if (np.linalg.det(mR[i]) <= 0): 
            return -1e6
        else:
            llik_total += (
                sc.special.loggamma(0.5*(dNu + iP)) - 
                sc.special.loggamma(0.5*dNu) 
                ) - 0.5 * np.log(np.linalg.det(mR[i])) - (
                0.5 * (dNu + iP) * np.log(1 + (mDataTilde[:,i] @ np.linalg.inv(mR[i]) @ mDataTilde[:,i]) / (dNu))
                )
                                    
    llik_marginal = iN * iP * (
            sc.special.loggamma(0.5*(dNu + 1)) - 
            sc.special.loggamma(0.5*dNu) 
            ) - (
            0.5 * (dNu + 1) * np.sum(
                np.log(1 + np.square(mDataTilde) / dNu)
            ))   

                
    llik_copula = llik_total - llik_marginal

    if np.isnan(llik_copula): 
        return -1e6
    else:
        return -llik_copula/iN
    
# =============================================================================
# 
# 
# def fStudentTCopulaLikelihood(mU, mR, dNu):
#     # Compute the Student's t(nu) copula likelihood function
#     #
#     # INPUTS:
#     # mU: a p x n matrix containing the data, formulated as PITs
#     # mR: a p x p correlation matrix
#     # dNu: a sccalar containing the degrees of freedom parameter
#     #
#     # OUTPUT:
#     # llik: a scalar containing the log copula likelihood
#     #
# 
#     # initialize the dimensions of the data
#     iP = mU.shape[0]
#     iN = mU.shape[1]
#     
#     # transform the PITs to pseudo Student t data
#     mDataTilde = t.ppf(mU, dNu)
#      
#     llik_copula = 0
#     for i in range(1, iN):
#         if (np.linalg.det(mR[i]) <= 0): 
#             return -1e6
#         else:
#             llik_copula += (
#                 sc.special.loggamma(0.5*(dNu + iP)) +
#                 (iP - 1) * sc.special.loggamma(0.5*dNu) -
#                 iP * sc.special.loggamma(0.5*(dNu + 1))
#                 
#                 ) - 0.5 * np.log(np.linalg.det(mR[i])) - (
#                 0.5 * (dNu + iP) * np.log(1 + (mDataTilde[:,i].T @ np.linalg.inv(mR[i]) @ mDataTilde[:,i]) / dNu)
#                 )
#                                     
# 
#     llik_copula = llik_copula - 0.5 * (dNu + 1) * np.sum(np.log(1 + np.square(mDataTilde) / dNu))
# 
#     if np.isnan(llik_copula): 
#         return -1e6
#     else:
#         return -llik_copula/iN
#     
# 
# 
#     
# =============================================================================
    
    
    