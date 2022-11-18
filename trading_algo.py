# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 23:57:55 2021

@author: Eigenaar
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
from itertools import groupby

from pull_data_yf import *
from filter_functions import *
np.set_printoptions(precision = 4)
plt.rcParams["figure.figsize"] = (10,6)

def get_average_lengths( arr ) :
    s = ''.join( ['0' if i < 1 else '1' for i in arr] )
    parts = s.split('0')
    return [len(p) for p in parts if len(p) > 0]

def fCopulaConditionalProbability(vU1, vU2, dNu, vRho):

    vEps1 = t.ppf(vU1, dNu)
    vEps2 = t.ppf(vU2, dNu)    
    
    return t.cdf( np.sqrt((dNu + 1)/(dNu + vEps2**2)) * ((vEps1 - vRho*vEps2)/np.sqrt(1 - vRho**2)), dNu + 1)
    
def fCalculateDailyDispersion(mEps, mPsi):

    vEps1 = mEps[0]
    vEps2 = mEps[1]
      
    vRho = mPsi[0]
    dNu = mPsi[1]
    
    vDaily_disp1 = fCopulaConditionalProbability(vEps1, vEps2, dNu, vRho) - 0.5
    vDaily_disp2 = fCopulaConditionalProbability(vEps2, vEps1, dNu, vRho) - 0.5
    
    return vDaily_disp1, vDaily_disp2
   
    
def fTradingBot(mEps, mPsi, dOpen, dStoploss, iT):
                              
    vDaily_disp1, vDaily_disp2 = fCalculateDailyDispersion(mEps, mPsi)
    
    vTotal_disp1 = np.zeros(iT)
    vTotal_disp2 = np.zeros(iT)

    mPosition = np.zeros((iT, 2))
    vTrade = np.zeros(iT)       
    iStop_loss_count = 0
    
    iTrigger = 0
    for i in range(1, iT):

        # Create mispricing index
        vTotal_disp1[i] = vTotal_disp1[i-1] + vDaily_disp1[i]
        vTotal_disp2[i] = vTotal_disp2[i-1] + vDaily_disp2[i]
                                   
        # Create mispricing index
        if mPosition[i-1, 0] != 0:
        
            if iTrigger == 1:
                if (vTotal_disp1[i] <= 0 or vTotal_disp1[i] >= dStoploss):
                    if vTotal_disp1[i] >= dStoploss:
                        iStop_loss_count += 1
                        
                    mPosition[i] = [0, 0]
                    vTotal_disp1[i], vTotal_disp2[i] = 0, 0
                    vTrade[i] = 1
                    
                    iTrigger = 0
                    
                else:
                    mPosition[i] = mPosition[i-1]

            elif iTrigger == 2:
                if (vTotal_disp2[i] >= 0 or vTotal_disp2[i] <= -dStoploss):
                                      
                    if vTotal_disp2[i] <= -dStoploss:
                        iStop_loss_count += 1
                    
                    mPosition[i] = [0, 0]
                    vTotal_disp1[i], vTotal_disp2[i] = 0, 0
                    vTrade[i] = 1
                    
                    iTrigger = 0       
                    
                else:
                    mPosition[i] = mPosition[i-1]
                    
            elif iTrigger == 3:
                if (vTotal_disp1[i] >= 0 or vTotal_disp1[i] <= -dStoploss):
                    if vTotal_disp1[i] <= -dStoploss:
                        iStop_loss_count += 1
                    
                    mPosition[i] = [0, 0]
                    vTotal_disp1[i], vTotal_disp2[i] = 0, 0
                    vTrade[i] = 1
                    
                    iTrigger = 0     
                                        
                else:
                    mPosition[i] = mPosition[i-1]
            
            elif iTrigger == 4:
                if (vTotal_disp2[i] >= 0 or vTotal_disp2[i] <= -dStoploss): 
                    
                    if vTotal_disp2[i] <= -dStoploss:
                        iStop_loss_count += 1
                    
                    mPosition[i] = [0, 0]
                    vTotal_disp1[i], vTotal_disp2[i] = 0, 0
                    vTrade[i] = 1
                    
                    iTrigger = 0    

                else:
                    mPosition[i] = mPosition[i-1]

            else:
                mPosition[i] = mPosition[i-1]

        else:        
            if vTotal_disp1[i] >= dOpen or vTotal_disp2[i] <= -dOpen:  
                mPosition[i] = [1 ,-1]    
                vTrade[i] = 1
                
                if vTotal_disp1[i] >= dOpen :
                    iTrigger = 1
                else: 
                    iTrigger = 2
                            
            elif vTotal_disp2[i] >= dOpen or vTotal_disp1[i] <= -dOpen:          
                mPosition[i] = [-1 , 1]       
                vTrade[i] = 1
                
                if vTotal_disp1[i] <= dOpen :
                    iTrigger = 3
                else: 
                    iTrigger = 4
                                
    return mPosition, vTrade, vTotal_disp1, vTotal_disp2, iStop_loss_count

def fTestStrategy(vTickers, mPrices, mReturns, mSig_hat, vTheta_hat_marginal, vTheta_hat_copula, vOptim_thresholds, iDays, dTrading_costs):

    mEps = fCalculateResiduals(mReturns, mSig_hat, vTheta_hat_marginal)
    mU = fCalculatePIT(mEps, vTheta_hat_marginal)
    
    if len(vTheta_hat_copula) > 3:
        mR = fFilterAdccCopula(vTheta_hat_copula, mEps)
    else:
        mR = fFilterDccCopula(vTheta_hat_copula, mEps)
    
    vRho = mR[:, 1, 0]
    vPsi_hat = [vRho, vTheta_hat_copula[-1]]
        
    mPosition, vTrade, vTotal_disp1, vTotal_disp2, iStop_loss_count = fTradingBot(mU, vPsi_hat, vOptim_thresholds[0], vOptim_thresholds[1], iDays)
        
    vPrices1 = mPrices[0]
    vPrices2 = mPrices[1]
    
    vReturns1 = mReturns[0]
    vReturns2 = mReturns[1]
          
    vRatio = vPrices1/vPrices2
    
    mTot_returns1 = mPosition[:,0] * (vReturns1)
    mTot_returns2 = mPosition[:,1] * (vReturns2)
    
    vLong_Short = (mTot_returns1 + mTot_returns2 - 2*vTrade*dTrading_costs*100)
    dPnL = np.sum(vLong_Short)
    
    dMean = np.mean(vLong_Short)
    dVol = np.var(vLong_Short)
    
    fAnnualised_returns = ((1 + dPnL/100)**(252/iDays) - 1)*100
    fAnnualised_volatility = np.sqrt(252*dVol)
    
    dSharpe_ratio = fAnnualised_returns/fAnnualised_volatility
    dMax_drawdown = np.min(np.cumsum(vLong_Short[5:]) - np.maximum.accumulate(np.cumsum(vLong_Short[5:])))
 
    vSignal = np.copy(vTrade)    
    vSignal[vSignal == 0] = np.nan
    vPosition_change = vSignal * mPosition[:,0]
    
    vLong_spread = vPosition_change == 1
    vShort_spread = vPosition_change == -1
    vUnwind_spread = vPosition_change == 0
            
    vLong_spread = vLong_spread * vRatio * vSignal
    vShort_spread = vShort_spread * vRatio * vSignal
    vUnwind_spread = vUnwind_spread * vRatio * vSignal
    
    vLong_spread[vLong_spread == 0] = np.nan
    vShort_spread[vShort_spread == 0] = np.nan
    vUnwind_spread[vUnwind_spread == 0] = np.nan
            
    fig, axs = plt.subplots(3, 1, sharex=True)

    axs[0].plot((vRatio[-380:]), color='black', label=vTickers[0]+'/'+vTickers[1])
    axs[0].plot((vUnwind_spread[-380:]), color='gray', marker='o', label = 'Unwind')    
    axs[0].plot((vLong_spread[-380:]), color='green', marker='^', label = 'Buy')
    axs[0].plot((vShort_spread[-380:]), color='red', marker='v', label = 'Sell')
    
    axs[1].plot(mPosition[-380:, 0], color='black', label='Position (Monetary Value)')

    axs[2].plot(vTotal_disp2[-380:], color='black', label='TI_'+vTickers[0])
    axs[2].plot(vTotal_disp1[-380:], color='red', label='TI_'+vTickers[1])
    
    axs[2].axhline(y=vOptim_thresholds[0], linestyle='--')
    axs[2].axhline(y=-vOptim_thresholds[0], linestyle='--')
    axs[2].axhline(y=vOptim_thresholds[1], linestyle='--')
    axs[2].axhline(y=-vOptim_thresholds[1], linestyle='--')
       
    axs[0].legend(loc=2)
    axs[1].legend(loc=2)
    axs[2].legend(loc=2)
    axs[2].legend(loc=2)

    fig.tight_layout(pad=1, w_pad=1, h_pad=1.0)    
    plt.show()

 
    vTrading_statistics = np.array([vOptim_thresholds[0], 
                                    vOptim_thresholds[1],
                                    np.sum(~np.isnan(vLong_spread))+np.sum(~np.isnan(vShort_spread)),
                                    np.sum(~np.isnan(vLong_spread))/(np.sum(~np.isnan(vLong_spread)) +np.sum(~np.isnan(vShort_spread))),
                                    np.mean(get_average_lengths(np.abs(mPosition[:,0]))),
                                    ])
    print(np.std(vLong_Short))
    vReturns_descriptive_statistics = np.array([dMean,
                              (dMean/np.std(vLong_Short))*np.sqrt(iDays),
                              dPnL/(np.sum(~np.isnan(vLong_spread))+np.sum(~np.isnan(vShort_spread))), 
                              fAnnualised_returns,
                              fAnnualised_volatility,
                              dSharpe_ratio,
                              dMax_drawdown
                              ])

    #print(vPerformance_metrics)
    print(vTrading_statistics)
    #dfReturns_descriptive_statistics = pd.DataFrame(vReturns_descriptive_statistics, columns=['# Trades', ' trade profitable', 'return profitability', 'cumulative retrun', 'Annualized Ret', 'annualized vol', 'Sharpe'])
    print(vReturns_descriptive_statistics)
    
    return  vLong_Short


def fOptimizeThresholds(vTickers, mPrices, mReturns, mSig_hat, mTheta_hat_marginals, vTheta_hat_copulas, iT, dTrading_costs):
        
    mEps = fCalculateResiduals(mReturns, mSig_hat, mTheta_hat_marginals)
    mU = fCalculatePIT(mEps, mTheta_hat_marginals)
    
    if len(vTheta_hat_copulas) > 3:
        mR = fFilterAdccCopula(vTheta_hat_copulas, mEps)
    else:
        mR = fFilterDccCopula(vTheta_hat_copulas, mEps)

    vRho = mR[:, 1, 0]
    vPsi_hat = [vRho, vTheta_hat_copulas[-1]]
    
    vOpen = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    vStoploss = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 1.9, 2]
    
    vPrices1 = mPrices[0]
    vPrices2 = mPrices[1]
    
    vReturns1 = mReturns[0]
    vReturns2 = mReturns[1]
  
    vBacktest_res = []
    for i in vOpen:
        for j in vStoploss:
            if j > i:    
                mPosition, vTrade, _, _, _ = fTradingBot(mU, vPsi_hat, i, j, iT)
                
                mTot_returns1 = mPosition[:,0] * (vReturns1/100)
                mTot_returns2 = mPosition[:,1] * (vReturns2/100)
                vPnL = (mTot_returns1 + mTot_returns2 - 2*vTrade*dTrading_costs)
                dPnL = np.sum(vPnL)
                
                fAnnualised_returns = (1 + dPnL)**(252/iT) - 1
                fAnnualised_volatility = np.sqrt(252*np.var(vPnL)) 
                
                #dSharpe_ratio = fAnnualised_returns/fAnnualised_volatility
                
                vBacktest_res.append([i, j, dPnL])
                
    dfBacktest = pd.DataFrame(vBacktest_res, columns=['Open Trigger', 'Stoploss', 'Return'])
    dfBest = dfBacktest.iloc[dfBacktest['Return'].argmax()]

    return dfBest


def fBacktestDCC(vTickers, mPrices, mReturns, vTheta_hat_marginals, vTheta_hat_copula, dTrading_costs, iSplit_years):
    
    # Split Data
    mIn_sample_prices, mOut_sample_prices = fSplitData(mPrices.values.T, iSplit_years)
    _, mValidation_prices = fSplitData(mIn_sample_prices, iSplit_years)
    
    mIn_sample_returns, mOut_sample_returns = fSplitData(mReturns, iSplit_years)
    _, mValidation_returns = fSplitData(mIn_sample_returns, iSplit_years)
    
    # Split Model Parameters
    vTheta_hat_gjr = vTheta_hat_marginals['GJR-GARCH']
    vTheta_hat_gas = vTheta_hat_marginals['GAS']
        
    vTheta_hat_gjr_dcc = vTheta_hat_copula['GJR-DCC']
    vTheta_hat_gas_dcc = vTheta_hat_copula['GAS-DCC']
    
    # Get Volatility Predictions
    dfUnivariate_predictions_backtesting = fGetUnivariatePredictions(vTickers, mValidation_returns, vTheta_hat_marginals)
    mSig_gjr_validation = dfUnivariate_predictions_backtesting['GJR-GARCH']
    mSig_gas_validation = dfUnivariate_predictions_backtesting['GAS']
    
    dfUnivariate_predictions_trading = fGetUnivariatePredictions(vTickers, mOut_sample_returns, vTheta_hat_marginals)
    mSig_gjr_test = dfUnivariate_predictions_trading['GJR-GARCH']
    mSig_gas_test = dfUnivariate_predictions_trading['GAS']
    
    # Start Backtests    
    iDays = 252*iSplit_years
    vOptim_thresholds_gjr = fOptimizeThresholds(vTickers, mValidation_prices, mValidation_returns, mSig_gjr_validation, vTheta_hat_gjr, vTheta_hat_gjr_dcc, iDays, dTrading_costs)
    vBacktest_res_gjr = fTestStrategy(vTickers, mOut_sample_prices, mOut_sample_returns, mSig_gjr_test, vTheta_hat_gjr, vTheta_hat_gjr_dcc, vOptim_thresholds_gjr, iDays, dTrading_costs)
  
    vOptim_thresholds_gas = fOptimizeThresholds(vTickers, mValidation_prices, mValidation_returns, mSig_gas_validation, vTheta_hat_gas, vTheta_hat_gas_dcc, iDays, dTrading_costs)
    vBacktest_res_gas = fTestStrategy(vTickers, mOut_sample_prices, mOut_sample_returns, mSig_gas_test, vTheta_hat_gas, vTheta_hat_gas_dcc, vOptim_thresholds_gas, iDays, dTrading_costs)
  
    #print(vOptim_thresholds_gjr)
    #print(vOptim_thresholds_gas)
        
    return {'GJR-DCC':vBacktest_res_gjr, 'GAS-DCC': vBacktest_res_gas}


def fBacktestADCC(vTickers, mPrices, mReturns, vTheta_hat_marginals, vTheta_hat_copula, dTrading_costs, iSplit_years):
    
    # Split Data
    mIn_sample_prices, mOut_sample_prices = fSplitData(mPrices.values.T, iSplit_years)
    _, mValidation_prices = fSplitData(mIn_sample_prices, iSplit_years)
    
    mIn_sample_returns, mOut_sample_returns = fSplitData(mReturns, iSplit_years)
    _, mValidation_returns = fSplitData(mIn_sample_returns, iSplit_years)

    vTheta_hat_gjr = vTheta_hat_marginals['GJR-GARCH']
    vTheta_hat_gas = vTheta_hat_marginals['GAS']

    vTheta_hat_gjr_adcc = vTheta_hat_copula['GJR-ADCC']
    vTheta_hat_gas_adcc = vTheta_hat_copula['GAS-ADCC']
    
    dfUnivariate_predictions_backtesting = fGetUnivariatePredictions(vTickers, mValidation_returns, vTheta_hat_marginals)
    mSig_gjr_validation = dfUnivariate_predictions_backtesting['GJR-GARCH']
    mSig_gas_validation = dfUnivariate_predictions_backtesting['GAS']
    
    dfUnivariate_predictions_trading = fGetUnivariatePredictions(vTickers, mOut_sample_returns, vTheta_hat_marginals)
    mSig_gjr_test = dfUnivariate_predictions_trading['GJR-GARCH']
    mSig_gas_test = dfUnivariate_predictions_trading['GAS']
        
    iDays = 252*iSplit_years
    vOptim_thresholds_gjr = fOptimizeThresholds(vTickers, mValidation_prices, mValidation_returns, mSig_gjr_validation, vTheta_hat_gjr, vTheta_hat_gjr_adcc, iDays, dTrading_costs)
    vOptim_thresholds_gas = fOptimizeThresholds(vTickers, mValidation_prices, mValidation_returns, mSig_gas_validation, vTheta_hat_gas, vTheta_hat_gas_adcc, iDays, dTrading_costs)
    
    vBacktest_res_gjr =  fTestStrategy(vTickers, mOut_sample_prices, mOut_sample_returns, mSig_gjr_test, vTheta_hat_gjr, vTheta_hat_gjr_adcc, vOptim_thresholds_gjr, iDays, dTrading_costs)
    vBacktest_res_gas =  fTestStrategy(vTickers, mOut_sample_prices, mOut_sample_returns, mSig_gas_test, vTheta_hat_gas, vTheta_hat_gas_adcc, vOptim_thresholds_gas, iDays, dTrading_costs)
        
    #print(vOptim_thresholds_gjr)
    #print(vOptim_thresholds_gas)
    
    return  {'GJR-ADCC': vBacktest_res_gjr, 'GAS-ADCC': vBacktest_res_gas}


def fTradingBotTemp(vEps1, vEps2, dfTheta, dOpen, dStoploss, iT):

    vDaily_disp1, vDaily_disp2 = fCalculateDailyDispersion(vEps1, vEps2, dfTheta)
    
    vTotal_disp1 = np.zeros(iT)
    vTotal_disp2 = np.zeros(iT)

    mPosition = np.zeros((iT, 2))
        
    iTrigger = 0
    for i in range(1, iT):

        # Create mispricing index
        vTotal_disp1[i] = vTotal_disp1[i-1] + vDaily_disp1[i]
        vTotal_disp2[i] = vTotal_disp2[i-1] + vDaily_disp2[i]
                                   
        # Create mispricing index
        if mPosition[i-1, 0] != 0:
        
            if iTrigger == 1:
                if (vTotal_disp1[i] <= 0 or vTotal_disp1[i] >= dStoploss):
                    mPosition[i] = [0, 0]
                    vTotal_disp1[i], vTotal_disp2[i] = 0, 0
                    iTrigger = 0
                    print('Day', i, 'CLOSE SPREAD')            
                else:
                    mPosition[i] = mPosition[i-1]
                    print('Day', i, 'HOLD SPREAD')
            
            elif iTrigger == 2:
                if (vTotal_disp2[i] >= 0 or vTotal_disp2[i] <= -dStoploss):
                    mPosition[i] = [0, 0]
                    vTotal_disp1[i], vTotal_disp2[i] = 0, 0
                    iTrigger = 0    
                    print('Day', i, 'CLOSE SPREAD')            
                else:
                    mPosition[i] = mPosition[i-1]
                    print('Day', i, 'HOLD SPREAD')
                    
            elif iTrigger == 3:
                if (vTotal_disp1[i] >= 0 or vTotal_disp1[i] <= -dStoploss):
                    mPosition[i] = [0, 0]
                    vTotal_disp1[i], vTotal_disp2[i] = 0, 0
                    iTrigger = 0    
                    print('Day', i, 'CLOSE SPREAD')      
                else:
                    mPosition[i] = mPosition[i-1]
                    print('Day', i, 'HOLD SPREAD')
            
            elif iTrigger == 4:
                if (vTotal_disp2[i] >= 0 or vTotal_disp2[i] <= -dStoploss):
                    mPosition[i] = [0, 0]
                    vTotal_disp1[i], vTotal_disp2[i] = 0, 0
                    iTrigger = 0    
                    print('Day', i, 'CLOSE SPREAD')            
                else:
                    mPosition[i] = mPosition[i-1]
                    print('Day', i, 'HOLD SPREAD')
            else:
                mPosition[i] = mPosition[i-1]
                print('Day', i, 'HOLD SPREAD')
             
        else:        
            if vTotal_disp1[i] >= dOpen or vTotal_disp2[i] <= -dOpen:  
                print('Day', i, "BUY SPREAD")
                mPosition[i] = [1 ,-1]     
                if vTotal_disp1[i] >= dOpen :
                    iTrigger = 1
                else: 
                    iTrigger = 2
                            
            elif vTotal_disp2[i] >= dOpen or vTotal_disp1[i] <= -dOpen:          
                print('Day', i, "SHORT SPREAD")
                mPosition[i] = [-1 , 1]       
                if vTotal_disp1[i] <= dOpen :
                    iTrigger = 3
                else: 
                    iTrigger = 4
            else:
                print('Day', i, "DO NOTHING")
    
    #print(np.sum(vTrade))              
    return mPosition


