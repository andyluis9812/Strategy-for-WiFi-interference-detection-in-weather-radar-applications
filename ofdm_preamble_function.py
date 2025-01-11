# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:39:49 2025

@author: andyl
"""

import numpy as np

def ofdm_preamble_function(Fs):
    """
    Function that generates the OFDM preamble at the sampling frequency specified by the parameter Fs.   
    Input parameters:
    Fs (int): Sampling frequency in MHz.
    
    Output parameters:
    np.array: OFDM preamble samples sampled at the specified frequency.
    """
    short_sym=np.sqrt(13/6)*np.array([0, 0, 1+1j, 0, 0, 0, -1-1j, 0, 0, 0, 1+1j, 0, 0, 0, -1-1j, 0, 0, 0, -1-1j, 0, 0, 0, 1+1j, 0, 0, 0, 0, 0, 0, 0, -1-1j, 0, 0, 0, -1-1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0, 0])
        
    large_sym=np.array([1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1 ,-1, 1, -1, 1, 1, 1, 1, 0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1])
        
    short_sym_f=(0+0j)*np.zeros(64)
    short_sym_f[0:27]=short_sym[26:]
    short_sym_f[-26:]=short_sym[0:26]
    
    large_sym_f=(0+0j)*np.zeros(64)
    large_sym_f[0:27]=large_sym[26:]
    large_sym_f[-26:]=large_sym[0:26]
    
    short_sym_t=np.fft.ifft(short_sym_f, 64)
    large_sym_t=np.fft.ifft(large_sym_f, 64)
    
    final_short_sym=short_sym_t[0:16]
    final_large_sym=large_sym_t
    cyclic_prefix=large_sym_t[-32:]
    
    preamble=np.concatenate((final_short_sym, final_short_sym, final_short_sym, final_short_sym, final_short_sym, final_short_sym, final_short_sym, final_short_sym, final_short_sym, final_short_sym, cyclic_prefix, final_large_sym, final_large_sym))
    
    return preamble[::int(np.floor(len(preamble)/(16*Fs)))]