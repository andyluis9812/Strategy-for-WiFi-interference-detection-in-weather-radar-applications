# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 09:58:35 2024

@author: andyl
"""

import numpy as np

def cfar_estimation_function(sequence, window, guard, separation):
    """    
    Function that estimates R*σ^4 from the output of the statistic. To estimate  
    R*σ^4 for sample k from the statistic output, samples separated by 'separation'  
    are selected from a window of length 'window', which is separated by 'guard'  
    samples from sample k 

    Input parameters:  
        sequence (np.array): Statistic  
        window (int): Number of samples in the window used for estimation  
        guard (int): Number of guard samples between sample k and the window  
        from which the estimation samples will be taken  
        separation (int): Separation between samples within the window  
            
    Output:  
        estimated_variance (np.array): Estimated values of R*σ^4 for each sample of the statistic 
    """
    estimated_variance=np.zeros(len(sequence))
    
    for i in np.arange(window+guard-separation, len(sequence)):
        #The samples are selected for the estimation
        aux=sequence[i-window-guard+separation:i-guard+1:separation]
        #The estimation is performed with the selected samples
        estimated_variance[i]=(1/len(aux))*np.sum(aux)
    return estimated_variance
