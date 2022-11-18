# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:30:16 2021

@author: PARAGUAS
"""

import numpy as np
import scipy as sc
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import t
from scipy.stats import norm
from scipy.linalg import sqrtm

from link_functions import *
from filter_functions import *
from llikelihood_functions import *
from display_estimation_results import *
from pull_data_yf import *
from trading_algo import *
from get_model_estimations import *
from plot_results import *

np.set_printoptions(precision = 4)
plt.rcParams["figure.figsize"] = (10,6)


def fCalculateTradingPerformance(mReturns, mPosition, vTrade, dTrading_fees):
    
    mTot_returns1 = mPosition[:,0] * (mReturns[0]/100)
    mTot_returns2 = mPosition[:,1] * (mReturns[1]/100)
    
    vPnL = mTot_returns1 + mTot_returns2 - 2*vTrade*dTrading_fees
    
    return vPnL


def fCalculateLambda(dfReturns, dfUnivariate_predictions, vTheta_hat_marginals):
    mSig_hat = dfUnivariate_predictions['GAS']
    mTheta_hat = vTheta_hat_marginals['GAS']
    
    mEps = fCalculateResiduals(dfReturns, mSig_hat, mTheta_hat)
    mEta = np.copy(mEps)
    mEta[mEta >= 0] = 0
        
    mQ_bar = np.corrcoef(mEps[0], mEps[1]) 
    mN_bar = np.corrcoef(mEta[0], mEta[1])
    
    mQ_bar_inv_sqrt = np.linalg.inv(sqrtm(mQ_bar))
    
    vEigen_values, vEigen_vectors = np.linalg.eig(mQ_bar_inv_sqrt @ mN_bar @ mQ_bar_inv_sqrt)
    
    #print(np.max(vEigen_values))
    
    
    
    
    
def fParseBacktestingParameters(vTickers, mPrices, mReturns, dfRes, dfBacktest_res_dcc, dfBacktest_res_adcc, iSplit_years, dTrading_fees):
    _, mOut_sample_returns = fSplitData(mReturns, iSplit_years)
    
    dfBacktest_res_gjr_dcc = dfBacktest_res_dcc['GJR-DCC']
    dfBacktest_res_gas_dcc = dfBacktest_res_dcc['GAS-DCC']
    
    dfBacktest_res_gjr_adcc = dfBacktest_res_adcc['GJR-DCC']
    dfBacktest_res_gas_adcc = dfBacktest_res_adcc['GAS-ADCC']
    

    vReturns_tilde1 = fCalculateTradingPerformance(mReturns, dfBacktest_res_gjr_dcc[0], dfBacktest_res_gjr_dcc[1], dTrading_fees)
    vReturns_tilde2 = fCalculateTradingPerformance(mReturns, dfBacktest_res_gas_dcc[0], dfBacktest_res_gas_dcc[1], dTrading_fees)
    
    vReturns_tilde3 = fCalculateTradingPerformance(mReturns, dfBacktest_res_gjr_adcc[0], dfBacktest_res_gjr_adcc[1], dTrading_fees)
    vReturns_tilde4 = fCalculateTradingPerformance(mReturns, dfBacktest_res_gas_adcc[0], dfBacktest_res_gas_adcc[1], dTrading_fees)
    
    fPlotTradingResults(mPrices, mReturns, mPosition, vTotal_disp1, vTotal_disp2, vTrade, dTrading_costs)
    fPlotTradingResults(mPrices, mReturns, mPosition, vTotal_disp1, vTotal_disp2, vTrade, dTrading_costs)
    fPlotTradingResults(mPrices, mReturns, mPosition, vTotal_disp1, vTotal_disp2, vTrade, dTrading_costs)
    fPlotTradingResults(mPrices, mReturns, mPosition, vTotal_disp1, vTotal_disp2, vTrade, dTrading_costs)

    
    printPerformanceMetrics(vReturns_tilde1, 'GJR-DCC')
    printPerformanceMetrics(vReturns_tilde2, 'GAS-DCC')
    printPerformanceMetrics(vReturns_tilde3, 'GJR-ADCC')
    printPerformanceMetrics(vReturns_tilde4, 'GAS-ADCC')
    
def fInitializeBacktestingNoCosts(vTickers, mPrices, mReturns, dfRes, iSplit_years, dTrading_fees):
                    
    dfResults_marginal_estimations = dfRes[0]
    dfResults_copula_estimations = dfRes[1]

    vTheta_hat_marginals = fUnpack_theta_hat_marginals(dfResults_marginal_estimations)      
    vTheta_hat_copula = fUnpack_theta_hat_copulas(dfResults_copula_estimations)

    dfBacktest_res_dcc =  fBacktestDCC(vTickers, mPrices, mReturns, vTheta_hat_marginals, vTheta_hat_copula,  dTrading_fees, iSplit_years)
    dfBacktest_res_adcc = fBacktestADCC(vTickers, mPrices, mReturns, vTheta_hat_marginals, vTheta_hat_copula, dTrading_fees, iSplit_years)


    fPlotCurves(dfBacktest_res_dcc, dfBacktest_res_adcc)
    #fParseBacktestingParameters(vTickers, mPrices, mReturns, dfRes, dfBacktest_res_dcc, dfBacktest_res_adcc, iSplit_years, dTrading_fees)
    
def fInitializeBacktesting(vTickers, mPrices, mReturns, dfRes, iSplit_years, dTrading_fees):
                    
    dfResults_marginal_estimations = dfRes[0]
    dfResults_copula_estimations = dfRes[1]

    vTheta_hat_marginals = fUnpack_theta_hat_marginals(dfResults_marginal_estimations)      
    vTheta_hat_copula = fUnpack_theta_hat_copulas(dfResults_copula_estimations)

    dfBacktest_res_dcc =  fBacktestDCC(vTickers, mPrices, mReturns, vTheta_hat_marginals, vTheta_hat_copula,  dTrading_fees, iSplit_years)
    dfBacktest_res_adcc = fBacktestADCC(vTickers, mPrices, mReturns, vTheta_hat_marginals, vTheta_hat_copula, dTrading_fees, iSplit_years)


    fPlotCurves(dfBacktest_res_dcc, dfBacktest_res_adcc)
    #fParseBacktestingParameters(vTickers, mPrices, mReturns, dfRes, dfBacktest_res_dcc, dfBacktest_res_adcc, iSplit_years, dTrading_fees)

def fFitModels(vTickers, mPrices, mReturns, dfTheta_star, iSplit_years, options):
    
    mIn_sample_prices, _ = fSplitData(mPrices.values.T, iSplit_years)
    mIn_sample_returns, _ = fSplitData(mReturns, iSplit_years)

    print(mIn_sample_returns.shape)
    
    #unpack theta star
    vTheta_star_garch = dfTheta_star['GARCH']
    vTheta_star_gjr_garch = dfTheta_star['GJR-GARCH']
    vTheta_star_gas = dfTheta_star['GAS']
    vTheta_star_dcc = dfTheta_star['DCC']
    vTheta_star_adcc = dfTheta_star['ADCC']

    # Step1: Estimate marginal models, use insample returns
    dfResults_marginal_estimations = fEstimateMarginals(mIn_sample_returns, vTheta_star_garch, vTheta_star_gjr_garch, vTheta_star_gas, options)
    fParseEstimationResultsMarginals(vTickers, mIn_sample_returns, dfResults_marginal_estimations)
    
    vTheta_hat_marginals = fUnpack_theta_hat_marginals(dfResults_marginal_estimations)
    dfUnivariate_predictions_train = fGetUnivariatePredictions(vTickers, mIn_sample_returns, vTheta_hat_marginals)
    
    # Step1.5
    #fTestMispecifications(vTickers, mIn_sample_returns, dfUnivariate_predictions_train, vTheta_hat_marginals)
    
    # Step2: Estimate copula models, use insample returns
    dfResults_copula_estimations = fCopulaEstimation(mIn_sample_returns, dfUnivariate_predictions_train, vTheta_hat_marginals, vTheta_star_dcc, vTheta_star_adcc, options)
    fParseEstimationResultsCopulas(vTickers, mReturns, vTheta_hat_marginals, dfResults_copula_estimations, dfResults_marginal_estimations)
      
    return [dfResults_marginal_estimations, dfResults_copula_estimations]

def fParseEstimationResultsMarginals(vTickers, dfReturns, tDict_results_marginal_estimations): 

    iT = np.shape(dfReturns)[1]
    vReturns1 = dfReturns[0]
    vReturns2 = dfReturns[1]
    
    dfResultsGarch = tDict_results_marginal_estimations['GARCH']
    dfResultsGjrGarch = tDict_results_marginal_estimations['GJR-GARCH']
    dfResultsGas = tDict_results_marginal_estimations['GAS']
    
    fCompareInformationCriteriaUnivariate(vTickers[0], dfResultsGarch[0], dfResultsGjrGarch[0], dfResultsGas[0], iT)
    fCompareInformationCriteriaUnivariate(vTickers[1], dfResultsGarch[1], dfResultsGjrGarch[1], dfResultsGas[1], iT)
        
    fGetparameterEstimatesUnivariate(vTickers[0], dfResultsGarch[0], dfResultsGjrGarch[0], dfResultsGas[0], iT)
    fGetparameterEstimatesUnivariate(vTickers[1], dfResultsGarch[1], dfResultsGjrGarch[1], dfResultsGas[1], iT)
    
def fParseEstimationResultsCopulas(vTickers, dfReturns, vTheta_hat_marginals, dfResults_copula_estimations, tDict_results_marginal_estimations):

    iT = np.shape(dfReturns)[1]
    dfUnivariate_predictions = fGetUnivariatePredictions(vTickers, dfReturns, vTheta_hat_marginals)
    fPlotFilteredVolatilties(dfReturns, dfUnivariate_predictions)

    fCalculateLambda(dfReturns, dfUnivariate_predictions, vTheta_hat_marginals)
    dfTheta_hat_copulas = fUnpack_theta_hat_copulas(dfResults_copula_estimations)

    dfDynamicCorrelations = fGetDynamicCorrelations(dfReturns, dfUnivariate_predictions, vTheta_hat_marginals, dfTheta_hat_copulas)
    fPlotFilteredConditionalCorrelation(dfDynamicCorrelations)
        
    fCompareInformationCriteriaCopula(vTickers, dfResults_copula_estimations, tDict_results_marginal_estimations, iT)
    fGetparameterEstimatesCopula(vTickers, dfResults_copula_estimations, iT)
    
def main():
 
    # Tickers
    vTickers_GOOG = ['GOOG', 'GOOGL']
    vTickers_BMW = ['BMW.DE', 'BMW3.DE']
    vTickers_BHP = ['BHP', 'BBL']
    vTickers_HEN = ['HEN.DE', 'HEN3.DE']
    vTickers_HEIC = ['HEI-A', 'HEI']
    vTickers_RDS = ['RDS-A', 'RDS-B']
    #vTickers_INX = ['^AEX', '^DAX']

    sPeriod = '15y'   
    vPeriod = ["2006-06-01", "2021-06-01"]
    
    print("====== PULLING DATA ======")
    
    #mPrices_GOOG = get_data(vTickers_GOOG, vPeriod)
    mPrices_BMW = get_data(vTickers_BMW, vPeriod)
    mPrices_BHP = get_data(vTickers_BHP, vPeriod)
    mPrices_HEN = get_data(vTickers_HEN, vPeriod)
    mPrices_HEIC = get_data(vTickers_HEIC, vPeriod)
    mPrices_RDS = get_data(vTickers_RDS, vPeriod)
    #mPrices_IND = get_data(vTickers_INX, vPeriod)
    
    
    
    print("====== CALCULATING RETRUNS ======")
    mReturns_BMW = get_returns(mPrices_BMW)
    mReturns_BHP = get_returns(mPrices_BHP)
    mReturns_HEN = get_returns(mPrices_HEN)
    mReturns_HEIC = get_returns(mPrices_HEIC)
    mReturns_RDS = get_returns(mPrices_RDS)
    #mReturns_IND = get_returns(mPrices_IND)
    
    plot_graphs_pair2(vTickers_BMW, mPrices_BMW, '-', vPeriod, 'BMW')
    plot_graphs_pair2(vTickers_HEN, mPrices_HEN, '-', vPeriod, 'Henkel')
    plot_graphs_pair2(vTickers_HEIC, mPrices_HEIC, '-', vPeriod, 'Heico')
    plot_graphs_pair2(vTickers_RDS, mPrices_RDS, '-', vPeriod, 'Shell')
        
    plot_graphs_pair(vTickers_BMW, mReturns_BMW, '.', vPeriod, 'BMW')
    plot_graphs_pair(vTickers_HEN, mReturns_HEN, '.', vPeriod, 'Henkel')
    plot_graphs_pair(vTickers_HEIC, mReturns_HEIC, '.', vPeriod, 'Heico')
    plot_graphs_pair(vTickers_RDS, mReturns_RDS, '.', vPeriod, 'Shell')
    
# =============================================================================
#     vPerformance_metrics = np.array([
#             np.mean(mReturns_BMW[0]),
#             np.sqrt(np.var(mReturns_BMW[0])),
#             np.min(mReturns_BMW[0]),
#             np.max(mReturns_BMW[0]),
#             sc.stats.skew(mReturns_BMW[0]),
#             sc.stats.kurtosis(mReturns_BMW[0])
#         ])
#     
#     print(vPerformance_metrics)
#     
#     vPerformance_metrics = np.array([
#             np.mean(mReturns_BMW[1]),
#             np.sqrt(np.var(mReturns_BMW[1])),
#             np.min(mReturns_BMW[1]),
#             np.max(mReturns_HEN[1]),
#             sc.stats.skew(mReturns_BMW[1]),
#             sc.stats.kurtosis(mReturns_BMW[1])
#         ])
#     
#     print(vPerformance_metrics)
#     
#     print(np.corrcoef(mReturns_BMW[0], mReturns_BMW[1]))
#     print(np.corrcoef(mReturns_HEIC[0], mReturns_HEIC[1]))
#     print(np.corrcoef(mReturns_HEN[0], mReturns_HEN[1]))
#     print(np.corrcoef(mReturns_RDS[0], mReturns_RDS[1]))
#     
# =============================================================================

    iSplit_years = 3
    dTrading_fees = 0.002
    
    dMu = 0.01    
    dOmega = 0.1
    dAlpha = 0.03
    dBeta = 0.93
    dGamma = 0.04
    dNu = 6
    
    dOmega_gas = 0.006
    dAlpha_gas = 0.055
    dBeta_gas = 0.99
    
    options={'disp':False, 'maxiter':5000}   
    vParams = [dMu, dOmega, dAlpha, dBeta, dGamma, dNu, dOmega_gas, dAlpha_gas, dBeta_gas]
    dfTheta_star = fDefineThetaStar(vParams)
       
    
    dfRes_BWM = fFitModels(vTickers_BMW, mPrices_BMW, mReturns_BMW, dfTheta_star, iSplit_years, options)
    fInitializeBacktesting(vTickers_BMW, mPrices_BMW, mReturns_BMW, dfRes_BWM, iSplit_years, dTrading_fees)
    fInitializeBacktestingNoCosts(vTickers_BMW, mPrices_BMW, mReturns_BMW, dfRes_BWM, iSplit_years, 0)

    dfRes_HEIC = fFitModels(vTickers_HEIC, mPrices_HEIC, mReturns_HEIC, dfTheta_star, iSplit_years, options)
    fInitializeBacktesting(vTickers_HEIC, mPrices_HEIC, mReturns_HEIC, dfRes_HEIC, iSplit_years, dTrading_fees)  
    fInitializeBacktestingNoCosts(vTickers_HEIC, mPrices_HEIC, mReturns_HEIC, dfRes_HEIC, iSplit_years, 0)  

    dfRes_HEN = fFitModels(vTickers_HEN, mPrices_HEN, mReturns_HEN, dfTheta_star, iSplit_years, options)
    fInitializeBacktesting(vTickers_HEN, mPrices_HEN, mReturns_HEN, dfRes_HEN, iSplit_years, dTrading_fees)
    fInitializeBacktestingNoCosts(vTickers_HEN, mPrices_HEN, mReturns_HEN, dfRes_HEN, iSplit_years, 0)

    dfRes_RDS = fFitModels(vTickers_RDS, mPrices_RDS, mReturns_RDS, dfTheta_star, iSplit_years, options)
    fInitializeBacktesting(vTickers_RDS, mPrices_RDS, mReturns_RDS, dfRes_RDS, iSplit_years, dTrading_fees)
    fInitializeBacktestingNoCosts(vTickers_RDS, mPrices_RDS, mReturns_RDS, dfRes_RDS, iSplit_years, 0)



###########################################################
### start main
if __name__ == "__main__":
    main()
    
    
    
    
    

    
