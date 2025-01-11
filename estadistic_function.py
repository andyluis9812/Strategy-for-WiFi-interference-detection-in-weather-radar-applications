# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 09:45:15 2024

@author: andyl
"""

import numpy as np

def estadistic_function(y, M, L):
    """
    Function that calculates the squared magnitude of the correlation between a window  
    of length M from the given signal and another window of length M from the same signal.  
    The separation between windows is given by L.  
    
    Input parameters:  
        y (list or np.array): Input signal  
        M (int): Window size  
        L (int): Distance between windows  
        
    Output:  
        estadistic (np.array): Statistic values 
    """
    y = np.array(y)     # Ensure the signal 'y' is a numpy array
    N = len(y)          # Length of the signal 'y'
    
    estadistic = np.zeros(N - M - L + 1)

    # Calculate the shifted autocorrelation
    for k in range(N - M - L + 1):
        y_window = y[k:k+M]
        y_shifted = y[k+L:k+L+M]
        estadistic[k] = (np.abs(np.sum(np.conjugate(y_window) * y_shifted)))**2
    
    return estadistic


