# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 16:07:59 2021

@author: PARAGUAS
"""

import numpy as np

def fParameterTransformUnivariateGAS(vTheta):
        r = np.array([
            vTheta[0],                                                      # dMu
            vTheta[1],                                                      # dOmega
            np.exp(vTheta[2])/(1 + np.exp(vTheta[2])),                      # dAlpha
            np.exp(vTheta[3])/(1 + np.exp(vTheta[3])),                      # dBeta
            2 + np.exp(vTheta[4])                                           # dNu
        
            ])

        
        return r

def fParameterTransformUnivariateGARCHt(vTheta):
        r = np.array([
            vTheta[0],                                                      # dMu
            np.exp(vTheta[1]),                                              # dOmega
            np.exp(vTheta[2])/(1 + np.exp(vTheta[2]) + np.exp(vTheta[3])),  # dAlpha
            np.exp(vTheta[3])/(1 + np.exp(vTheta[2]) + np.exp(vTheta[3])),  # dBeta
            2 + np.exp(vTheta[4])   
            ])


        
        return r
        
def fParameterTransformUnivariateGJRGARCHt(vTheta):
        r = np.array([
            
            vTheta[0],                                                      # dMu
            np.exp(vTheta[1]),                                              # dOmega
            np.exp(vTheta[2])/(1 + np.exp(vTheta[2]) + np.exp(vTheta[3]) + np.exp(vTheta[4])),  # dAlpha
            np.exp(vTheta[3])/(1 + np.exp(vTheta[2]) + np.exp(vTheta[3]) + np.exp(vTheta[4])),  # dBeta
            2*np.exp(vTheta[4])/(1 + np.exp(vTheta[2]) + np.exp(vTheta[3]) + np.exp(vTheta[4])),  # dBeta
            2 + np.exp(vTheta[5]) 
            
            ])

        
        
        return r
    
def fParameterTransformCopulaDCC(vTheta):

    if len(vTheta) > 2:
    
        r = np.array([
            np.exp(vTheta[0])/(1 + np.exp(vTheta[0]) + np.exp(vTheta[1])),
            np.exp(vTheta[1])/(1 + np.exp(vTheta[0]) + np.exp(vTheta[1])),  
            np.exp(vTheta[2])         
            ])
   
        
            
    else:
        r = (
            np.exp(vTheta[0])/(1 + np.exp(vTheta[0]) + np.exp(vTheta[1])),
            np.exp(vTheta[1])/(1 + np.exp(vTheta[0]) + np.exp(vTheta[1])), 
          )
    
    return r

def fParameterTransformCopulaADCC(vTheta):

    if len(vTheta) > 2:
    
        r = np.array([ 
            np.exp(vTheta[0])/(1 + np.exp(vTheta[0]) + np.exp(vTheta[1]) + np.exp(vTheta[2])),
            np.exp(vTheta[1])/(1 + np.exp(vTheta[0]) + np.exp(vTheta[1]) + np.exp(vTheta[2])),  
            np.exp(vTheta[2])/(1 + np.exp(vTheta[0]) + np.exp(vTheta[1]) + np.exp(vTheta[2])),  
            np.exp(vTheta[3])
            ])
 
    
    else:
        r = ( 
            np.exp(vTheta[0])/(1 + np.exp(vTheta[0]) + np.exp(vTheta[1])),
            np.exp(vTheta[1])/(1 + np.exp(vTheta[0]) + np.exp(vTheta[1])), 
          )
    
    return r





def fDefineThetaStar(vParams):
    
    #dMu, dOmega, dAlpha, dBeta, dGamma, dNu, dOmega_copula, dAlpha_copula, dBeta_copula, dNu_copula = vParams
    dMu, dOmega, dAlpha, dBeta, dGamma, dNu, dOmega_gas, dAlpha_gas, dBeta_gas  = vParams
    
    vTheta_star_garch = [dMu, 
                         np.log(dOmega), 
                         np.log(dAlpha/(1 - dAlpha - dBeta)), 
                         np.log(dBeta/(1 - dAlpha - dBeta)), 
                         np.log(dNu - 2)]
    
    vTheta_star_gjr_garch = [dMu, 
                             np.log(dOmega), 
                             np.log(dAlpha/(1 - dAlpha - dBeta - dGamma/2)), 
                             np.log(dBeta/(1 - dAlpha - dBeta - dGamma/2)), 
                             np.log((dGamma/2)/(1 - dAlpha - dBeta - dGamma/2)), 
                             np.log(dNu - 2)]
    vTheta_star_gas = [dMu, 
                       dOmega_gas, 
                       np.log(dAlpha_gas/(1 - dAlpha_gas)), 
                       np.log(dBeta_gas/(1 - dBeta_gas)),  
                       np.log(dNu - 2)]
        
    vTheta_star_dcc = [np.log(dAlpha/(1 - dAlpha - dBeta)), 
                       np.log(dBeta/(1 - dAlpha - dBeta)),  
                       np.log(dNu)]
    
    vTheta_star_adcc = [np.log(dAlpha/(1 - dAlpha - dBeta - dGamma/2)), 
                       np.log(dBeta/(1 - dAlpha - dBeta - dGamma/2)), 
                       np.log((dGamma/2)/(1 - dAlpha - dBeta - dGamma/2)), 
                       np.log(dNu)]

    return{'GARCH':vTheta_star_garch, 'GJR-GARCH':vTheta_star_gjr_garch, 'GAS':vTheta_star_gas, 'DCC':vTheta_star_dcc, 'ADCC':vTheta_star_adcc}


def fUnpack_theta_hat_marginals(tDict_results_marginal_estimations):
            
    dfResults_garch = tDict_results_marginal_estimations['GARCH']
    dfResults_gjr_garch = tDict_results_marginal_estimations['GJR-GARCH']
    dfResults_gas = tDict_results_marginal_estimations['GAS']
    
    vTheta_garch1 = fParameterTransformUnivariateGARCHt(dfResults_garch[0].x)
    vTheta_garch2 = fParameterTransformUnivariateGARCHt(dfResults_garch[1].x)
        
    vTheta_gjr_garch1 = fParameterTransformUnivariateGJRGARCHt(dfResults_gjr_garch[0].x)
    vTheta_gjr_garch2 = fParameterTransformUnivariateGJRGARCHt(dfResults_gjr_garch[1].x)
    
    vTheta_gas1 = fParameterTransformUnivariateGAS(dfResults_gas[0].x) 
    vTheta_gas2 = fParameterTransformUnivariateGAS(dfResults_gas[1].x) 

    return {'GARCH':[vTheta_garch1, vTheta_garch2], 
            'GJR-GARCH':[vTheta_gjr_garch1, vTheta_gjr_garch2], 
            'GAS': [vTheta_gas1, vTheta_gas2]}

def fUnpack_theta_hat_copulas(dfResults_copula_estimations ):
            
    dfResults_gjr_dcc = dfResults_copula_estimations['GJR-DCC']
    dfResults_gas_dcc = dfResults_copula_estimations['GAS-DCC']
    
    dfResults_gjr_adcc = dfResults_copula_estimations['GJR-ADCC']
    dfResults_gas_adcc = dfResults_copula_estimations['GAS-ADCC']
    
    vTheta_hat_gjr_dcc = fParameterTransformCopulaDCC(dfResults_gjr_dcc.x)
    vTheta_hat_gas_dcc = fParameterTransformCopulaDCC(dfResults_gas_dcc.x)
    
    vTheta_hat_gjr_adcc = fParameterTransformCopulaADCC(dfResults_gjr_adcc.x)
    vTheta_hat_gas_adcc = fParameterTransformCopulaADCC(dfResults_gas_adcc.x)
    
    return {'GJR-DCC':vTheta_hat_gjr_dcc, 
            'GJR-ADCC':vTheta_hat_gjr_adcc, 
            'GAS-DCC': vTheta_hat_gas_dcc,
            'GAS-ADCC': vTheta_hat_gas_adcc,
            }