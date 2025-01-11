# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:33:05 2024

@author: andyl
"""
import numpy as np

def threshold_function(pfa, estimation, N):
    """
    Function that calculates the detection threshold for each sample of the statistic  
    based on the established false alarm probability, the number of samples  
    used for CFAR estimation, and the CFAR estimation for each sample of the statistic  
    
    Input parameters:  
        pfa (int): False alarm probability  
        estimation (np.array): Values of the R*σ^4 estimation for each sample of the statistic  
        N (int): Number of samples used for CFAR estimation 
    
    Output:  
        threshold (np.array): Detection threshold  
    """
    threshold=np.zeros(len(estimation))
    
    for i in np.arange(0, len(threshold)):
        threshold[i]=N*(((pfa)**(-1/N))-1)*estimation[i]
    
    return threshold
