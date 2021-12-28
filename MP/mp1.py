#!/usr/bin/env python
# coding: utf-8

# In[52]:


# Import functions and libraries
import numpy as np
from numpy import r_, exp, cos, sin, pi, zeros, ones, hanning, sqrt, log, floor, reshape, mean
from scipy import signal
from numpy.fft import fft


data_path = 'data.csv'
data = np.genfromtxt(data_path, delimiter=',')
Ns = len(data[:,0])
fs = 100  # [Hz]
dt = 1/fs  # unit time
t = np.arange(0,dt*Ns,dt)
Acc = np.sqrt(np.square(data[:,0])+np.square(data[:,1])+np.square(data[:,2]))

fc = 3.5 #cuttoff freq [Hz]
b,a = signal.butter(10, fc/(fs/2), btype='lowpass', analog=False)
Acc_LPF = signal.filtfilt(b,a,Acc)

width = 275  #number of samples for each window
iter = (len(Acc)//width)+1
peak_dynamic_temp=[]
Avg=0
for i in range(iter): 
    Avg = np.average(Acc_LPF[0+i*width:width+i*width])

    prom_dynamic = 4/(np.std(Acc_LPF[0+i*width:width+i*width]))
    peak_dynamic, _ = signal.find_peaks(Acc_LPF[0+i*width:width+i*width], prominence=prom_dynamic)

    peak_dynamic_temp = np.append(peak_dynamic_temp, peak_dynamic+i*width)


peak_dynamic = peak_dynamic_temp.astype(int)

with open('result.txt', 'w') as f:
    f.write('%d' %len(peak_dynamic))

