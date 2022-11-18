# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 12:27:54 2021

@author: Eigenaar
"""
import numpy as np
import scipy as sc
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from pull_data_yf import *
plt.rcParams["figure.figsize"] = (10,6)

def fPlotInitialData(vTickers, mPrices, mReturns):
    
    plt.plot(mPrices[0], label='Prices ' + vTickers[0])
    plt.plot(mPrices[1], label='Prices ' + vTickers[1])
    plt.legend(fontsize=20)
    plt.show()
    
    

def plot_graphs_pair(vTickers, vData, marker, vPeriod, sTitle):

    vData1 = vData[0]
    vData2 = vData[1]
    
    start = pd.Timestamp(vPeriod[0])
    end = pd.Timestamp(vPeriod[1])
    t = np.linspace(start.value, end.value, len(vData1))
    t = pd.to_datetime(t)
    
    plt.plot(t, vData1, marker, color = 'black',  label=vTickers[0])
    plt.plot(t, vData2, marker, color = 'red', label=vTickers[1])
    plt.legend(loc=2, fontsize=12)
    plt.ylabel('Return', fontsize=12)
    plt.tight_layout()
        
    plt.show()

def plot_graphs_pair2(vTickers,dfData, marker, vPeriod, sTitle):
    vData = dfData.values.T
    vData1 = vData[0]
    vData2 = vData[1]
    
    start = pd.Timestamp(vPeriod[0])
    end = pd.Timestamp(vPeriod[1])
    t = np.linspace(start.value, end.value, len(vData1))
    t = pd.to_datetime(t)
    
    plt.plot(t, vData1, marker, color = 'black',  label=vTickers[0])
    plt.plot(t, vData2, marker, color = 'red', label=vTickers[1])
    plt.legend(loc=2, fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.tight_layout()
    
    plt.show()

       

def fPlotFilteredVolatilties(mIn_sample_returns, dfUnivariate_predictions):
        
    
    start = pd.Timestamp('2006-06-01')
    end = pd.Timestamp('2021-06-01')
    
    vTime = np.linspace(start.value, end.value, len(mIn_sample_returns[0]))
    vTime = pd.to_datetime(vTime)
        
    # Plot filtered volatilities    
    plt.plot(vTime, np.abs(mIn_sample_returns[0]), '.', label='Abs return')
    plt.plot(vTime, (dfUnivariate_predictions['GJR-GARCH'][0]), color = 'red', label='GJR-t-GARCH')
    plt.plot(vTime, (dfUnivariate_predictions['GAS'][0]), '--', color = 'black', label='t-GAS')
  
    plt.legend(loc=2, fontsize=12)
    plt.tight_layout()
    plt.show()


def fPlotFilteredConditionalCorrelation(dfDynamicCorrelations):
    
    #plt.hist(dfDynamicCorrelations['GAS-DCC'][:, 1, 0])
    #plt.hist(dfDynamicCorrelations['GAS-ADCC'][:, 1, 0])
    #plt.show()
    
    
    start = pd.Timestamp('2019-12-01')
    end = pd.Timestamp('2021-06-01')
    
    vTime = np.linspace(start.value, end.value, 380)
    vTime = pd.to_datetime(vTime)
           

    # Plot filtered correlations    
    plt.plot(vTime, dfDynamicCorrelations['GJR-DCC'][-380:, 1, 0], color = 'red',  label='Copula-DCC-GJR')
    plt.plot(vTime, dfDynamicCorrelations['GAS-DCC'][-380:, 1, 0], ':', color = 'black', label='Copula-DCC-GAS')
    plt.plot(vTime, dfDynamicCorrelations['GJR-ADCC'][-380:, 1, 0],'-.', color = 'navy',  label='Copula-ADCC-GJR')
    plt.plot(vTime, dfDynamicCorrelations['GAS-ADCC'][-380:, 1, 0], '--', color = 'limegreen', label='Copula-ADCC-GAS')
    
    
    plt.legend(loc=2, fontsize=12)
    plt.tight_layout()
    plt.show()



def fPlotCurves(dfBacktest_res_dcc, dfBacktest_res_adcc):
    
    start = pd.Timestamp('2018-06-01')
    end = pd.Timestamp('2021-06-01')
    
    vTime = np.linspace(start.value, end.value, len(dfBacktest_res_dcc['GJR-DCC']))
    vTime = pd.to_datetime(vTime)
        
    plt.plot(vTime, np.cumsum(dfBacktest_res_dcc['GJR-DCC']), color = 'maroon', label='Copula-DCC-GJR')    
    plt.plot(vTime, np.cumsum(dfBacktest_res_dcc['GAS-DCC']), ':', color = 'black',  label='Copula-DCC-GAS') 
    plt.plot(vTime, np.cumsum(dfBacktest_res_adcc['GJR-ADCC']), '-.', color = 'darkblue', label='Copula-ADCC-GJR')
    plt.plot(vTime, np.cumsum(dfBacktest_res_adcc['GAS-ADCC']), '--', color = 'darkgreen', label='Copula-ADCC-GAS')
    
    plt.legend(loc=2, fontsize=12)
    plt.tight_layout()
    plt.show()




def fPlotTradingResults(mPrices, mReturns, mPosition, vTotal_disp1, vTotal_disp2, vTrade, dTrading_costs):
    
    vPrices1 = mPrices[0]
    vPrices2 = mPrices[1]
    
    vReturns1 = mReturns[0]
    vReturns2 = mReturns[1]

    vRatio = vPrices1/vPrices2           
    mTot_returns1 = mPosition[:,0] * (vReturns1/100)
    mTot_returns2 = vRatio*mPosition[:,1] * (vReturns2/100)

    vLong_Short = (mTot_returns1 + mTot_returns2 - 2*vTrade*dTrading_costs)*100
    vSpread = mPrices[0]/mPrices[1]
        
    fig, axs = plt.subplots(2, 2, sharex=True)
    fig.suptitle(vTickers[0]+'/'+vTickers[1])
    axs[0,0].plot(vSpread)
    axs[0,0].grid()
    
    axs[1,0].plot(mPosition[:,0])
    axs[1,0].grid()
    
    axs[0,1].plot(np.cumsum(vLong_Short))
    axs[0,1].grid()
    
    axs[1,1].plot(vTotal_disp1)
    axs[1,1].plot(vTotal_disp2)
    axs[1,1].grid()

    fig.tight_layout(pad=1, w_pad=1, h_pad=1.0)    
    plt.show()
    
    
def fPlotInitialData(vTickers, mPrices, mReturns):
    
    plt.plot(mPrices[0], label='Prices ' + vTickers[0])
    plt.plot(mPrices[1], label='Prices ' + vTickers[1])
    plt.legend(fontsize=20)
    plt.show()