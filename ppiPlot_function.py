# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:13:19 2025

@author: andyl
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import numpy.matlib
warnings.filterwarnings("ignore")
import matplotlib.patches as patches

def ppiPlot_function(data, r, az, rect, x_origin=None, y_origin=None, x_large=None, y_large=None):
    
    """
    This function generates PPI (Plan Position Indicator) plots for the reflectivity estimation of a Doppler polarimetric radar.
    
    Input parameters:  
        data (np.array): Estimated reflectivity matrix. (Dimensions: 'number of azimuths' x 'range estimates').
        'r' (np.array): [1 x M] vector of range dimensions (distance to the radar), in this case in km.
        'az' (np.array): [1 x A] vector of averaged azimuth values.
        'rect' (bool): Parameter to indicate whether the rectangle is drawn or not. 
        'x_origin' (float): Value of the x-axis at the bottom-left vertex of the rectangle.
        'y_origin' (float): Value of the y-axis at the bottom-left vertex of the rectangle.
        'x_large' (float): Length of the rectangle along the x-axis.
        'y_large' (float): Length of the rectangle along the y-axis.
    Output:  
        PPI plot for reflectivity
     
    """
    
    if not isinstance(rect, bool):
        raise TypeError("The input parameter rect must be of type bool.")
    R1, AZ = np.meshgrid(r, az)
    X = R1 * np.sin(AZ * np.pi/180)
    Y = R1 * np.cos(AZ * np.pi/180)
    
    map_colors_ref = np.array([
        [128, 128, 128],
        [0, 236, 236],
        [1, 160, 246],
        [0, 0, 246],
        [0, 255, 0],
        [0, 200, 0],
        [0, 144, 0],
        [255, 255, 0],
        [231, 192, 0],
        [255, 144, 0],
        [255, 0, 0],
        [214, 0, 0],
        [192, 0, 0],
        [255, 0, 255],
        [153, 85, 201],
        [120, 100, 201]
    ]) / 255.0  
    
    custom_cmap = mpl.colors.ListedColormap(map_colors_ref)
    fig , ax = plt.subplots(figsize=(20, 16))
    mesh = plt.pcolormesh(X, Y, data.T, cmap = custom_cmap, shading = 'auto')
    plt.clim(0, 80)
    cbar = plt.colorbar(mesh)
    cbar.set_label('Reflectivity [dBZ]', fontsize=70)
    ax.set_xlabel('Range [km]', fontsize=70)
    ax.set_ylabel('Range [km]', fontsize=70)
    
    if rect==True:
        if not all(isinstance(parameters, float) for parameters in (x_origin, y_origin, x_large, y_large)):
            raise TypeError("The input parameters x_origin, y_origin, x_large, and y_large must be of type float.")
        rect = patches.Rectangle((x_origin, y_origin), x_large, y_large, linewidth=4, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
            
    plt.show()
    
    