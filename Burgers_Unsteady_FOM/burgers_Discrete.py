# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 11:29:47 2022

@author: rahul
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import savemat,loadmat



nx = 20
nt = 100
mu = 0.05

L = np.linspace(0, 1, nx).reshape(-1, 1)

time = np.linspace(0, 0.5, nt).reshape(-1, 1)

u0 = np.sin((np.pi)*L)

dx = L[1]-L[0]
dt = time[1]-time[0]

# Get the Matrices for the computation

#(1/dt)*(utp1-ut) = Au+F(u)

# Form A matrix

A =  np.zeros((nx, nx))


for i in range(1,nx-1):
    if (i == 1) :
        A[i,i+1] = 1
    elif (i == nx-2) :
        A[i,i-1] = 1
    else:
        A[i,i+1]=1
        A[i,i-1]=1
    A[i,i] = - 2

# Construct the Nonlinear Matrix

def F(nx,u):
    
    F = np.zeros((nx, nx))

    for i in range(1,nx-1):
        if (i == nx-2) :
           F[i,i] = -u[i]
        else:
           F[i,i+1]= u[i]
           F[i,i]=  -u[i]
    return F
        
const = (mu/(dx*dx))

u = u0
T = 0

u_Mat = None

for t in range(int(nt)):
    print("T=",T)
    T = T+dt
    u_updated =  u + const*dt*np.matmul(A,u) - (dt/dx)*np.matmul(F(nx,u),u)
    
    if u_Mat is None:
        u_Mat = u[:,0].reshape((u.shape[0], 1))
    else:
        u_Mat = np.append(u_Mat,u[:,0].reshape((u.shape[0], 1)),axis =1) 
    
    u = u_updated 
    
plt.plot(np.arange(0,nx),u0, label = 'inital')
plt.plot(np.arange(0,nx),u, label='final')

mdic = {"u_Mat": u_Mat ,"L":L, "time":time}
savemat("Burgers_FOM.mat", mdic)
        

        

