# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 20:33:03 2021

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

from link_functions import *
from filter_functions import *
from llikelihood_functions import *

def pull_yahoo_finance(sTicker, vPeriod):

    dfData = yf.Ticker(sTicker)
    
    dfPx_data = dfData.history(start=vPeriod[0], end=vPeriod[1])

    dfPx_data.index = dfPx_data.index.strftime('%Y-%m-%d')
    dfPx_close = dfPx_data['Close']
    dfPx_close.rename(sTicker, inplace=True)
    
    return dfPx_close

def get_data(vTickers, sPeriod):
    
    dfPx_close_dual_listings = pd.DataFrame()
    
    for i in vTickers:
        dfPx_close_dual_listings[i] = (pull_yahoo_finance(i, sPeriod))
        
    return dfPx_close_dual_listings
    

def get_returns(dfPrices):

    dfReturns = (np.diff(np.log(dfPrices.values.T))*100)
    
    return dfReturns


def fParsePrices(mTickers, sPeriod):

    mPrices = []

    for i in mTickers:

        mPrices.append(get_data(i, sPeriod))

    return mPrices

def fSplitData(mReturns, iSplit_year):
      
    iT = len(mReturns[0])
    iTrading_days = 252*iSplit_year
        
    mOut_sample = mReturns[:, -iTrading_days:]
    mIn_sample = mReturns[:, :(iT-iTrading_days)]
        
    
    
    return mIn_sample, mOut_sample



