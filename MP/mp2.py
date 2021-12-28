#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from numpy import r_, exp, cos, sin, pi, zeros, ones, hanning, sqrt, log, floor, reshape, mean
from scipy import signal
from numpy.fft import fft

# data import by readfile
data_path = 'data.csv'
data = np.genfromtxt(data_path, delimiter=',')
Ns = len(data[1:,0])
fs = 100  # [Hz]
dt = 1/fs  # unit time

# Initial orientation calculation
v2_raw = data[0,:]
v2 = v2_raw/np.linalg.norm(v2_raw) 

y=sqrt(1-np.square(v2[1]))
x=-v2[0]*v2[1]/y
z=x*v2[2]/v2[0]

v1 = np.zeros(3)
v1 = [x,y,z]

v0 = np.cross(v1,v2)

X = np.vstack((v0,v1,v2)).T
v0_g = np.array([1,0,0])
v1_g = np.array([0,1,0])
v2_g = np.array([0,0,1])
X_g = np.vstack((v0_g,v1_g,v2_g)).T

R = np.matmul(X_g,np.linalg.inv(X))

out1 = np.matmul(R,v0_g.T)


# 3D orientation tracking

def getdR(R0,dt,l):
    dth = np.linalg.norm(l,2)*dt    # Rotation Angle
    l_norm = l/np.linalg.norm(l)
    l_g = np.matmul(R0,l_norm.T)
    u_x = np.array([[0,-l_g[2], l_g[1]]
                    ,[l_g[2], 0, -l_g[0]]
                    ,[-l_g[1], l_g[0], 0]])  # cross product matrix
    dR = np.identity(3)*np.cos(dth) + np.sin(dth)*u_x + (1-np.cos(dth))*np.outer(l_g,l_g)
    return dR
    
R_now = R
for i in range(Ns):
    dR = getdR(R_now,dt,data[i+1,:])
    R_now = np.matmul(dR,R_now)
out2 = np.matmul(R_now,v0_g.T)


# Write file 
with open('result.txt', 'w') as f:
    f.write('    '.join(str(i) for i in out1)+'\n')
    f.write('    '.join(str(i) for i in out2))


# In[ ]:




