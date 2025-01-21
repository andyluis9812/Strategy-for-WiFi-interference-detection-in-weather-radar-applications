# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 08:21:17 2024

@author: andyl
"""
'''
This code is organized into cells. To continue, each cell is described below.

--------Import libraries cell:
    In this cell, the libraries used in the code are imported.
--------Import data cell:
    In this cell, the data from the weather radars used in this project is imported.
    It is important to clarify that we do not have permission to share the data 
    from the Argentinian weather radars. Therefore, this data is not available 
    to run this code.
--------Graphics cell:
    In this cell, we generate several graphs presented in the paper. Specifically, we 
    produce the PPI graphs using data from the RMA1 weather radar located in Córdoba City. 
    Additionally, we create a graph showing the in-phase component of two WiFi packets acquired 
    by this radar, as well as a graph of the detection statistic used in this project for these 
    two WiFi packets. Furthermore, we generate a graph illustrating the periodic structure of the 
    OFDM preamble from one of these WiFi packets."
--------OFDM Preamble cell:
    In this cell, the OFDM preamble is generated. This preamble, is used for all the theoretical 
    analysis conducted in this project.
--------Matched Filter cell:
    In this cell, we pass the sequence of short symbols from the preamble, obtained in the OFDM 
    Preamble cell, through the matched filter. Finally, we sample the output of the matched filter at 4 MHz.
--------Detection Probability cell:
    In this section, we obtain the theoretical curves for different amounts of samples of the detection 
    statistic used in the L*σ^4 estimation, as well as the ideal curve for the detection probability. Then, 
    we plot these curves.
--------Estimated detection probability cell:
    In this cell, we estimate the detection probability for N=10 and plot both the theoretical curve and the 
    estimated curve of the detection probability for N=10, as well as the ideal curve of the detection probability.
--------Calculation of the SNR of the packets cell:
    In this cell, we calculate the SNR of the WiFi packets acquired from the RMA1 weather radar and select the 
    packets with the best SNR. Then, we multiply these WiFi packets by constants to achieve a desired SNR in all 
    of them. Finally, we plot the SNR curve of these WiFi packets.
--------Location of interference over noise cell:
    In this cell, we take the WiFi packets obtained in the previous cell and add them to the data acquired from 
    the RMA6 weather radar, located in Mar del Plata city. These WiFi packets are inserted into a region where only 
    noise is present. Additionally, the resulting PPI graph is shown.
--------Apply the complete algorithm to the data contaminated with interference over noise cell:
    In this cell, we apply the designed algorithm to the data obtained in the previous cell. The resulting PPI graph 
    is also shown. Additionally, the detection probability is estimated.
--------Location of the interference over the phenomenon cell:
    In this cell, we take the WiFi packets obtained in the 'Calculation of the SNR of the Packets' cell and add them 
    to the data acquired from the RMA6 weather radar, located in Mar del Plata city. These WiFi packets are inserted 
    into a region where there are weather phenomena along with noise. The resulting PPI graph is also shown.  
--------Apply the complete algorithm to the data contaminated with interference over the phenomenon cell:
    In this cell, we apply the designed algorithm to the data obtained in the previous cell. The resulting PPI graph 
    is also shown. Additionally, the detection probability is estimated.
'''

#%% Import libraries

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import ncx2
from scipy.io import loadmat
import pandas as pd
import scipy.io

import delay_correlate_function
import ofdm_preamble_function
import ppiPlot_function
import threshold_function 
import estadistic_function 
import cfar_estimation_function 

#%% Import data

# Path from which the data is loaded
directory = 'D:/Materias/Doctorado/Delay and Correlate/Paper/Codigo/'

############Dataset from the RMA1 weather radar.
file_name1 = 'mis_variables'    #Name of the file that contains the dataset

# Load the dataset 
dict_data = loadmat(directory + file_name1) 

dataIQcpi_paq = np.array(dict_data['dataIQcpi'])            #Variable that contains the I&Q data from the RMA1 weather radar
rangeVect_paq = np.array(dict_data['rangeVect'])            #Variable that contains the range vector used in the RMA1 weather radar data
paqWifiRMA1 = np.array(dict_data['paqWifiRMA1'])            #Variable that contains information about the location of the WiFi packets
azimutAngProm_paq = np.array(dict_data['azimutAngProm'])    #Variable that contains the azimuth vector used in the RMA1 weather radar data

############Dataset from the RMA6 weather radar.
file_name2 = 'RMA06_modificada_sin_interferencia_IQ'

# Load the dataset
dict_data = loadmat(directory + file_name2) 

dataIQcpi = np.array(dict_data['RMA06_modificada_sin_interferencia_IQ'])    #Variable that contains the I&Q data from the RMA6 weather radar

file_name3 = 'Datos_reales_RMA06'

# Load the dataset
dict_data = loadmat(directory + file_name3) 

rangeVect = np.array(dict_data['rangeVect'])            #Variable that contains the range vector used in the RMA6 weather radar data
azimutAngProm = np.array(dict_data['azimutAngProm'])    #Variable that contains the azimuth vector used in the RMA6 weather radar data
num_cpi=int(np.array(dict_data['nroCPIs']))             #Variable that contains the number of CPIs of the RMA6 weather radar data
num_pulses=int(np.array(dict_data['cpi']))              #Variable that contains the number of pulses of each CPI in the RMA6 weather radar data

# Path where the results are saved
directory_results='D:/Materias/Doctorado/Delay and Correlate/Paper/Codigo/Resultados/'

#%%Graphics

nr_CPIs=azimutAngProm_paq.shape[1]
nr_RangeCell=rangeVect_paq.shape[1]
nr_Pulses=54

mod_pow = np.zeros((dataIQcpi_paq.shape[0], dataIQcpi_paq.shape[2]))

#Calculate the power
for i in np.arange(0, dataIQcpi_paq.shape[2]):
    for j in np.arange(0, dataIQcpi_paq.shape[0]):
        correlation = np.correlate(dataIQcpi_paq[j, :, i][:], dataIQcpi_paq[j, :, i][:], 'full')/nr_Pulses
        mod_pow [j, i] = correlation[int((len(correlation)-1)/2)]

#Arrange the range vector to multiply it by the power matrix

new_rangeVect=np.zeros(rangeVect_paq.shape[1])
for i in np.arange(0, rangeVect_paq.shape[1]):
    new_rangeVect[i]=rangeVect_paq[0,i]/1000

final_rangeVect=np.zeros((rangeVect_paq.shape[1], nr_CPIs))
for i in np.arange(0, nr_CPIs):
    final_rangeVect[:, i]=new_rangeVect[:]

final_rangeVect=final_rangeVect**2

# Calculate the reflectivity
ZedBZ_modified=35+10*np.log10(final_rangeVect)+10*np.log10(mod_pow)

#Plot the reflectivity of data with WiFi interference
plt.rcParams.update({
    'text.usetex':True,
    'font.family':'roman',     
    'xtick.labelsize': 70,      
    'ytick.labelsize': 70})
ppiPlot_function.ppiPlot_function(ZedBZ_modified, rangeVect_paq/1000, azimutAngProm_paq, False)

plt.rcParams['text.usetex']=False

M=20            #The repetition interval length
L=20            #The separation between two adjacent intervals


pulse=dataIQcpi_paq[:, 29, 15]          #Example pulse

#The distance vector is defined

distance_vector=np.zeros(len(pulse))    

for i in np.arange(0, len(pulse)):
    distance_vector[i]=rangeVect_paq[0, i]/1000

#Plot the in-phase component of the example pulse
plt.rcParams.update({
    'text.usetex':True,
    'font.family':'roman'})
plt.figure(figsize=(19, 11))
plt.plot(distance_vector, np.real(pulse), color='blue')
plt.xlim([50, 120])
plt.ylim([-0.005, 0.005])
plt.ylabel('In-phase component \n[AU]', fontsize=70)
plt.xlabel('Range [km]', fontsize=70)
plt.tight_layout()
plt.tick_params(axis='x', pad=15)  
plt.tick_params(axis='y', pad=15)  
plt.tick_params(labelsize=70)
plt.grid()
plt.show()

plt.rcParams['text.usetex']=False

#The example pulse statistic is calculated
estadistic=(np.abs(delay_correlate_function.delay_correlate_function(pulse, M, L)))**2

#Plot the stadistic of the example pulse
plt.rcParams.update({
    'text.usetex':True,
    'font.family':'roman'})
plt.figure(figsize=(19, 11))
plt.plot(distance_vector[0:-2*L+1], estadistic, linewidth=3, color='blue')
plt.xlim([50, 120])
plt.ylim([0, 1.05*10**(-8)])
plt.ylabel('Correlation squared \nmodulus [AU]', fontsize=70, labelpad=225)
plt.xlabel('Range [km]', fontsize=70)
plt.gca().yaxis.get_offset_text().set_fontsize(70)
plt.tick_params(axis='x', pad=15)  
plt.tick_params(axis='y', pad=15)  
plt.tick_params(labelsize=70)
plt.grid()
plt.show()

plt.rcParams['text.usetex']=False

# Plot the OFDM preamble of the in-phase component of the example pulse
plt.rcParams.update({
    'text.usetex':True,
    'font.family':'roman'})
plt.figure(figsize=(19, 11))
plt.plot(distance_vector, np.real(pulse), linewidth=3, color='blue')
plt.xlim([distance_vector[1965], distance_vector[2055]])
plt.vlines(x=[distance_vector[1970], distance_vector[1990], distance_vector[2010], distance_vector[2018], distance_vector[2034], distance_vector[2050]], ymin=-1, ymax=1, colors=['black', 'black', 'black', 'red', 'red', 'red'], linestyles='--', linewidth=3)
plt.ylim([-0.005, 0.005])
plt.ylabel('In-phase component \n[AU]', fontsize=70)
plt.xlabel('Range [km]', fontsize=70)
plt.tick_params(labelsize=70)
plt.grid()
plt.show()

plt.rcParams['text.usetex']=False

#%%OFDM Preamble

#Generate the OFDM preamble for a 20 MHz bandwidth channel
preamble=ofdm_preamble_function.ofdm_preamble_function()

#Sequence of short symbols.
secuence_short_sym=preamble[:160]
#%%Matched Filter

#Marched Filter
matched_filter=np.ones(20)

filter_output=np.convolve(np.concatenate((secuence_short_sym, secuence_short_sym)), matched_filter, mode='same')

#Output of the matched filter
final_filter_output=filter_output[80:-80]

#Output of the matched filter sampled at 4 MHz
final_filter_output_4M=final_filter_output[::5]
#%%Detection Probability

M=16            #The repetition interval length
L=16            #The separation between two adjacent intervals
P_FA = 1e-4     #False alarm probability

def integrand(u):
    """
    Function that returns the integrand of the integral present in the expression for the theoretical detection probability 
    
    Input parameters:
    u (float): Variable over which the integration is performed.
    
    Output parameters:
    float: Numerical value of the integrand for the specified input value
    """
    Q_value = 1-ncx2.cdf(2 * u * (P_FA**(-1 / N) - 1) / (2 * ENR / (L) + 1), 2, (2 * ENR**2 / L) / ((2 * ENR) / (L) + 1))
    return Q_value * u**(N - 1) * np.exp(-u)

realizations=1000       #Number of realizations       
                          
matrix=np.zeros(realizations,  dtype=complex)         

snr_vector=np.arange(2, 80, 0.5)    #SNR vector 
snr_vector_db=10*np.log10(snr_vector)    #SNR vector in dB

ideal_dp=np.zeros(len(snr_vector))

# Calculate the ideal detection probability
for i in np.arange(0, len(snr_vector)):
    ideal_dp[i]=1-ncx2.cdf((-2*np.log(P_FA))/((2*snr_vector[i]/M)+1), 2, (snr_vector[i]/(np.sqrt(snr_vector[i]+M/2)))**2)

# Calculate the theoretical detection probability for N=10
D_P_1=np.zeros(len(snr_vector))
N=10
for i in np.arange(0, len(snr_vector)):
    ENR=snr_vector[i]
    integral_result, _ = quad(integrand, 0, np.inf)
    factor = 1 / np.math.factorial(N-1)
    D_P_1[i] = factor * integral_result

# Calculate the theoretical detection probability for N=30
D_P_2=np.zeros(len(snr_vector))
N=30
for i in np.arange(0, len(snr_vector)):
    ENR=snr_vector[i]
    integral_result, _ = quad(integrand, 0, np.inf)
    factor = 1 / np.math.factorial(N-1)
    D_P_2[i] = factor * integral_result

# Calculate the theoretical detection probability for N=60
D_P_3=np.zeros(len(snr_vector))
N = 60
for i in np.arange(0, len(snr_vector)):
    ENR=snr_vector[i]
    integral_result, _ = quad(integrand, 0, np.inf)
    factor = 1 / np.math.factorial(N-1)
    D_P_3[i] = factor * integral_result

#Plot the ideal detection probability and the theoretical detection probability for N=10, N=30 and N=60
plt.rcParams.update({
    'text.usetex':True,
    'font.family':'roman'})
plt.figure(figsize=(19, 11))
plt.plot(snr_vector_db, D_P_1, label='N=10', linewidth=5, color='blue')
plt.plot(snr_vector_db, D_P_2, label='N=30', linewidth=5, color= 'red')
plt.plot(snr_vector_db, D_P_3, label='N=60', linewidth=5, color='green')
plt.plot(snr_vector_db, ideal_dp, label='Ideal', linestyle='--', color='black', linewidth=5)
plt.legend(fontsize=55)
plt.xlabel('ENR [dB]', fontsize=70)
plt.ylabel('Detection probability', fontsize=70, labelpad=100)
plt.grid()
plt.tick_params(labelsize=70)
plt.show()

plt.rcParams['text.usetex']=False

#%%Estimated detection probability

M=16            #The repetition interval length
L=16            #The separation between two adjacent intervals
P_FA = 1e-4     #False alarm probability

snr_vector=np.arange(2, 80, 0.5)    #SNR vector 
snr_vector_db=10*np.log10(snr_vector)    #SNR vector in dB

signal_energy=np.sum((np.abs(final_filter_output_4M))**2)/2     #Signal energy

variances=np.zeros(len(snr_vector))

for i in np.arange(0, len(variances)):
    variances[i]=signal_energy/snr_vector[i]

dp_cfar_realizations=1000      

cfar_matrix=np.zeros((dp_cfar_realizations, len(variances)),  dtype=complex)           #Matriz donde en cada fila se almaceanará los resultados del delay and correlate
dp_cfar_detections=np.zeros(len(variances))     

final_filter_output_4M_matrix=np.tile(final_filter_output_4M, (dp_cfar_realizations, 1))

N=10                #Number of samples used in the estimation
separation=32       #Separation between the samples used in the estimation       

alfa=N*(((P_FA)**(-1/N))-1)     #Calculate alfa

#Estimate the detection probability

for j in np.arange(0, len(variances)):
    print(j)
    real_noise=np.random.normal(0, np.sqrt(variances[j]/2), (dp_cfar_realizations, N*separation+2*L))
    imaginary_noise=np.random.normal(0, np.sqrt(variances[j]/2), (dp_cfar_realizations, N*separation+2*L))
    noise=real_noise+1j*imaginary_noise
    signal=final_filter_output_4M_matrix+noise[:, -2*L:]
    
    statistic=(np.abs(np.sum(np.conj(signal[:, :M]) * signal[:, M:], axis=1)))**2 
    
    noise_stadistic = (np.abs(np.apply_along_axis(delay_correlate_function.delay_correlate_function, axis=1, arr=noise, M=M, L=L)))**2
        
    estimated_variance=(1/N)*np.sum(noise_stadistic[:, :N*separation:separation], axis=1)
    
    threshold=alfa*estimated_variance
    
    dp_cfar_detections[j] = np.sum(statistic >= threshold)

detection_probability_cfar=dp_cfar_detections/dp_cfar_realizations

#Plot the ideal detection probability, the theoretical detection probability for N=10 and the estimated detection probability for N=10
plt.rcParams.update({
    'text.usetex':True,
    'font.family':'roman'})
plt.figure(figsize=(19, 11))
plt.plot(snr_vector_db, D_P_1, label='Theoretical', linewidth=5, color='blue')
plt.plot(snr_vector_db, detection_probability_cfar, label='Estimated', linestyle='--', color='red', linewidth=5)
plt.plot(snr_vector_db, ideal_dp, label='Ideal', linestyle='--', color='black', linewidth=5)
plt.legend(fontsize=55)
plt.xlabel('ENR [dB]', fontsize=70)
plt.ylabel('Detection probability', fontsize=70, labelpad=100)
plt.grid()
plt.tick_params(labelsize=70)
plt.show()

plt.rcParams['text.usetex']=False

#%%Calculation of the SNR of the packets

paq_Pow=np.zeros(len(paqWifiRMA1))

#Calculate the power of the WiFi packets
for i in np.arange(0, len(paqWifiRMA1)):
    paq_Pow[i]=np.mean((np.abs(paqWifiRMA1[i, 4]))**2)

paq_Pow_db=10*np.log10(paq_Pow)
paq_Pow_Max=np.max(paq_Pow)

# Packet indices
ind_Paq_1=np.arange(965, 1047)
ind_Paq_2=np.arange(1065, 1211)

ind=np.concatenate((ind_Paq_1, ind_Paq_2))

#The power of the packets of interest is corrected
for i in ind:
    paqWifiRMA1[i, 4]=paqWifiRMA1[i, 4]*np.sqrt(paq_Pow_Max/paq_Pow[i])
    paq_Pow[i]=np.mean((np.abs(paqWifiRMA1[i, 4]))**2)

paq_Pow_db=10*np.log10(paq_Pow)

#Plot the corrected SNR of the WiFi packets
plt.figure(figsize=(18, 14))
plt.title('SNR', fontsize=70)
plt.plot(paq_Pow_db)
plt.xlabel('Index', fontsize=70)
plt.ylabel('SNR [dB]', fontsize=70)
plt.grid()

#%%Location of interference over noise

#############Variables to configure#############
cpi_inicial=20      #CPI where the interference will start to appear
cpi_final=65        #CPI where the interference will stop appearing

numRangeCell=4849           #Number of range cells
cpi=54                      #Number of pulses per CPI
numCPIs=362                 #Numjbers of CPIs

#Variables to track the packets
ubicacion_paq=np.array([])
pulse_paq=np.array([])
cpi_paq=np.array([])
pulse=0
paq_count=0 

# Create an empty DataFrame
data = {
    'Number': [],
    'CPI': [],
    'Pulse': [], 
    'Sample': []
}

df = pd.DataFrame(data)

cpiInd=cpi_inicial

while paq_count<len(ind) and cpiInd<cpi_final:
    aux=np.random.randint(-300, 300)
    margin=2300+aux
    pulse=0
    while dataIQcpi.shape[0]-margin>int(paqWifiRMA1[int(ind[paq_count]), 3]-paqWifiRMA1[int(ind[paq_count]), 2]) and paq_count<len(ind):
        dataIQcpi[margin:margin+int(paqWifiRMA1[int(ind[paq_count]), 3])-int(paqWifiRMA1[int(ind[paq_count]), 2])+1, pulse, cpiInd]=dataIQcpi[margin:margin+int(paqWifiRMA1[int(ind[paq_count]), 3])-int(paqWifiRMA1[int(ind[paq_count]), 2])+1, pulse, cpiInd]+paqWifiRMA1[int(ind[paq_count]), 4].reshape(-1)
        df = df.append({'Number': ind[paq_count], 'CPI': cpiInd, 'Pulse': pulse , 'Sample': margin}, ignore_index=True)
        margin=margin+int(paqWifiRMA1[int(ind[paq_count]), 3]-paqWifiRMA1[int(ind[paq_count]), 2])
        paq_count=paq_count+1
        if paq_count==len(ind):
            break
        aux=np.random.randint(1, 5)
        pulse=pulse+aux
    if cpiInd % 7 == 0:
        cpiInd += 3
    else:
        cpiInd += 1
    
#The packet location data is saved
df.to_csv(directory_results+'Packets location.csv', index=False)
df.to_excel(directory_results+'Packets location.xlsx', index=False)

mod_pow = np.zeros((dataIQcpi.shape[0], dataIQcpi.shape[2]))

#Calculate the power
for i in np.arange(0, dataIQcpi.shape[2]):
    for j in np.arange(0, dataIQcpi.shape[0]):
        correlation = np.correlate(dataIQcpi[j, :, i][:], dataIQcpi[j, :, i][:], 'full')/cpi
        mod_pow [j, i] = correlation[int((len(correlation)-1)/2)]

#Arrange the range vector to multiply it by the power matrix

new_rangeVect=np.zeros(rangeVect.shape[1])
for i in np.arange(0, rangeVect.shape[1]):
    new_rangeVect[i]=rangeVect[0,i]/1000

final_rangeVect=np.zeros((rangeVect.shape[1], numCPIs))
for i in np.arange(0, numCPIs):
    final_rangeVect[:, i]=new_rangeVect[:]

final_rangeVect=final_rangeVect**2

# Calculate the reflectivity
ZedBZ_modified=35+10*np.log10(final_rangeVect)+10*np.log10(mod_pow)

#Save the reflectivity values
scipy.io.savemat(directory_results+'reflectivity', {'reflectivity': ZedBZ_modified})

#Save the IQ data with interference
scipy.io.savemat(directory_results+'IQ_data', {'datosIQ': dataIQcpi})


#Variables to configure the rectangle's position on the graph.
x_origin_n=-145.0
y_origin_n=-50.0
x_large_n=90.0
y_large_n=115.0

#Plot the reflectivity of data contaminated with interference over noise
plt.rcParams.update({
    'text.usetex':True,
    'font.family':'roman',     
    'xtick.labelsize': 70,      
    'ytick.labelsize': 70})

ppiPlot_function.ppiPlot_function(ZedBZ_modified, rangeVect/1000, azimutAngProm, True, x_origin_n, y_origin_n, x_large_n, y_large_n)

plt.rcParams['text.usetex']=False

#%%Apply the complete algorithm to the data contaminated with interference over noise
#############Variables to configure#############
M=20            #The repetition interval length
L=20            #The separation between two adjacent intervals
P_FA = 1e-9     #False alarm probability

#Parameters for estimation
window=440             #Number of samples of the window used for estimation
guard=40               #Number of guard samples between the sample of interest and the estimation window
separation=40          #Separation between the samples used in the estimation

N=window/separation    #Number of samples used in the estimation

detections=np.zeros((dataIQcpi.shape[0]-2*L+1, dataIQcpi.shape[1], dataIQcpi.shape[2]))     

for i in np.arange(0, dataIQcpi.shape[2]):
    print(i)
    for j in np.arange(0, dataIQcpi.shape[1]):
        pulse=dataIQcpi[:, j, i]
        d_c=estadistic_function.estadistic_function(pulse, M, L)
        estimation=cfar_estimation_function.cfar_estimation_function(d_c, window, guard, separation)
        detect_threshold=threshold_function.threshold_function(P_FA, estimation, N)
        detections[:, j, i]=d_c>detect_threshold

detections_count=0

search_range=10

for i in np.arange(0, len(df)):
    paq_df=int(df.loc[i, 'Number'])
    sample_df=int(df.loc[i, 'Sample'])
    cpi_df=int(df.loc[i, 'CPI'])
    pulse_df=int(df.loc[i, 'Pulse'])
    if int(np.sum(detections[sample_df-search_range:sample_df+search_range, pulse_df, cpi_df]))>=1:
        detections_count=detections_count+1
        len_p=len(paqWifiRMA1[int(paq_df), 4])
        dataIQcpi[sample_df:sample_df+len_p, pulse_df, cpi_df]=np.zeros(len_p)       
detection_probability=detections_count/len(df)

mod_pow = np.zeros((dataIQcpi.shape[0], dataIQcpi.shape[2]))

#Calculate the power
for i in np.arange(0, dataIQcpi.shape[2]):
    for j in np.arange(0, dataIQcpi.shape[0]):
        correlation = np.correlate(dataIQcpi[j, :, i][:], dataIQcpi[j, :, i][:], 'full')/cpi
        mod_pow [j, i] = correlation[int((len(correlation)-1)/2)]

#Arrange the range vector to multiply it by the power matrix

new_rangeVect=np.zeros(rangeVect.shape[1])
for i in np.arange(0, rangeVect.shape[1]):
    new_rangeVect[i]=rangeVect[0,i]/1000

final_rangeVect=np.zeros((rangeVect.shape[1], numCPIs))
for i in np.arange(0, numCPIs):
    final_rangeVect[:, i]=new_rangeVect[:]

final_rangeVect=final_rangeVect**2

# Calculate the reflectivity
ZedBZ_modified=35+10*np.log10(final_rangeVect)+10*np.log10(mod_pow)

#Plot the reflectivity of data contaminated with interference over noise
plt.rcParams.update({
    'text.usetex':True,
    'font.family':'roman',     
    'xtick.labelsize': 70,      
    'ytick.labelsize': 70})
ppiPlot_function.ppiPlot_function(ZedBZ_modified, rangeVect/1000, azimutAngProm, False)

plt.rcParams['text.usetex']=False

#%%Location of the interference over the phenomenon

#############Variables to configure#############
cpi_inicial=160      #CPI where the interference will start to appear

numRangeCell=4849           #Number of range cells
cpi=54                      #Number of pulses per CPI
numCPIs=362                 #Numjbers of CPIs

#Variables to track the packets
ubicacion_paq=np.array([])
pulse_paq=np.array([])
cpi_paq=np.array([])
pulse=0
paq_count=0 

# Create an empty DataFrame
data = {
    'Number': [],
    'CPI': [],
    'Pulse': [], 
    'Sample': []
}

df = pd.DataFrame(data)

cpiInd=cpi_inicial

while paq_count<len(ind):
    aux=np.random.randint(-300, 300)
    margin=2300+aux
    pulse=0
    while dataIQcpi.shape[0]-margin>int(paqWifiRMA1[int(ind[paq_count]), 3]-paqWifiRMA1[int(ind[paq_count]), 2]) and paq_count<len(ind):
        dataIQcpi[margin:margin+int(paqWifiRMA1[int(ind[paq_count]), 3])-int(paqWifiRMA1[int(ind[paq_count]), 2])+1, pulse, cpiInd]=dataIQcpi[margin:margin+int(paqWifiRMA1[int(ind[paq_count]), 3])-int(paqWifiRMA1[int(ind[paq_count]), 2])+1, pulse, cpiInd]+paqWifiRMA1[int(ind[paq_count]), 4].reshape(-1)
        df = df.append({'Number': ind[paq_count], 'CPI': cpiInd, 'Pulse': pulse , 'Sample': margin}, ignore_index=True)
        margin=margin+int(paqWifiRMA1[int(ind[paq_count]), 3]-paqWifiRMA1[int(ind[paq_count]), 2])
        paq_count=paq_count+1
        if paq_count==len(ind):
            break
        aux=np.random.randint(1, 5)
        pulse=pulse+aux
    if cpiInd % 7 == 0:
        cpiInd += 3
    else:
        cpiInd += 1
    
#The packet location data is saved
df.to_csv(directory_results+'Packets location.csv', index=False)
df.to_excel(directory_results+'Packets location.xlsx', index=False)

mod_pow = np.zeros((dataIQcpi.shape[0], dataIQcpi.shape[2]))

#Calculate the power
for i in np.arange(0, dataIQcpi.shape[2]):
    for j in np.arange(0, dataIQcpi.shape[0]):
        correlation = np.correlate(dataIQcpi[j, :, i][:], dataIQcpi[j, :, i][:], 'full')/cpi
        mod_pow [j, i] = correlation[int((len(correlation)-1)/2)]

#Arrange the range vector to multiply it by the power matrix

new_rangeVect=np.zeros(rangeVect.shape[1])
for i in np.arange(0, rangeVect.shape[1]):
    new_rangeVect[i]=rangeVect[0,i]/1000

final_rangeVect=np.zeros((rangeVect.shape[1], numCPIs))
for i in np.arange(0, numCPIs):
    final_rangeVect[:, i]=new_rangeVect[:]

final_rangeVect=final_rangeVect**2

# Calculate the reflectivity
ZedBZ_modified=35+10*np.log10(final_rangeVect)+10*np.log10(mod_pow)

#Save the reflectivity values
scipy.io.savemat(directory_results+'reflectivity', {'reflectivity': ZedBZ_modified})

#Save the IQ data with interference
scipy.io.savemat(directory_results+'IQ_data', {'datosIQ': dataIQcpi})

#Variables to configure the rectangle's position on the graph.
x_origin_p=35.0
y_origin_p=5.0
x_large_p=110.0
y_large_p=122.0

#Plot the reflectivity of data contaminated with interference over the phenomenon
plt.rcParams.update({
    'text.usetex':True,
    'font.family':'roman',     
    'xtick.labelsize': 70,      
    'ytick.labelsize': 70})
ppiPlot_function.ppiPlot_function(ZedBZ_modified, rangeVect/1000, azimutAngProm, True, x_origin_p, y_origin_p, x_large_p, y_large_p)

plt.rcParams['text.usetex']=False

#%%Apply the complete algorithm to the data contaminated with interference over the phenomenon
#############Variables to configure#############
M=20            #The repetition interval length
L=20            #The separation between two adjacent intervals
P_FA = 1e-9     #False alarm probability

#Parameters for estimation
window=440             #Number of samples of the window used for estimation
guard=40               #Number of guard samples between the sample of interest and the estimation window
separation=40          #Separation between the samples used in the estimation

N=window/separation    #Number of samples used in the estimation

detections=np.zeros((dataIQcpi.shape[0]-2*L+1, dataIQcpi.shape[1], dataIQcpi.shape[2]))     

for i in np.arange(0, dataIQcpi.shape[2]):
    print(i)
    for j in np.arange(0, dataIQcpi.shape[1]):
        pulse=dataIQcpi[:, j, i]
        d_c=estadistic_function.estadistic_function(pulse, M, L)
        estimation=cfar_estimation_function.cfar_estimation_function(d_c, window, guard, separation)
        detect_threshold=threshold_function.threshold_function(P_FA, estimation, N)
        detections[:, j, i]=d_c>detect_threshold

detections_count=0

search_range=10

for i in np.arange(0, len(df)):
    paq_df=int(df.loc[i, 'Number'])
    sample_df=int(df.loc[i, 'Sample'])
    cpi_df=int(df.loc[i, 'CPI'])
    pulse_df=int(df.loc[i, 'Pulse'])
    if int(np.sum(detections[sample_df-search_range:sample_df+search_range, pulse_df, cpi_df]))>=1:
        detections_count=detections_count+1
        len_p=len(paqWifiRMA1[int(paq_df), 4])
        dataIQcpi[sample_df:sample_df+len_p, pulse_df, cpi_df]=np.zeros(len_p)       
detection_probability=detections_count/len(df)

mod_pow = np.zeros((dataIQcpi.shape[0], dataIQcpi.shape[2]))

#Calculate the power
for i in np.arange(0, dataIQcpi.shape[2]):
    for j in np.arange(0, dataIQcpi.shape[0]):
        correlation = np.correlate(dataIQcpi[j, :, i][:], dataIQcpi[j, :, i][:], 'full')/cpi
        mod_pow [j, i] = correlation[int((len(correlation)-1)/2)]

#Arrange the range vector to multiply it by the power matrix

new_rangeVect=np.zeros(rangeVect.shape[1])
for i in np.arange(0, rangeVect.shape[1]):
    new_rangeVect[i]=rangeVect[0,i]/1000

final_rangeVect=np.zeros((rangeVect.shape[1], numCPIs))
for i in np.arange(0, numCPIs):
    final_rangeVect[:, i]=new_rangeVect[:]

final_rangeVect=final_rangeVect**2

# Calculate the reflectivity
ZedBZ_modified=35+10*np.log10(final_rangeVect)+10*np.log10(mod_pow)

#Plot the reflectivity of data contaminated with interference over the phenomenon
plt.rcParams.update({
    'text.usetex':True,
    'font.family':'roman',     
    'xtick.labelsize': 70,      
    'ytick.labelsize': 70})
ppiPlot_function.ppiPlot_function(ZedBZ_modified, rangeVect/1000, azimutAngProm, False)

plt.rcParams['text.usetex']=False



