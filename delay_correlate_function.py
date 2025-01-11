# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:29:47 2025

@author: andyl
"""

import numpy as np

def delay_correlate_function(y, M, L):
    """
    Function that calculates the lagged autocorrelation of the signal y.
    
    Input parameters:
        y (list o np.array): Input signal.
        M (int): Window size.
        L (int): Distance between windows.
    
    Output parameters:
        correlation (np.array): Delayed autocorrelation values for each k.
    """
    y = np.array(y) 
    N = len(y)
    
    correlation = np.zeros(N - M - L + 1, dtype=np.complex)

    for k in range(N - M - L + 1):
        y_window = y[k:k+M]
        y_shifted = y[k+L:k+L+M]
        
        correlation[k] = np.sum(np.conjugate(y_window) * y_shifted)
    return correlation