# -*- coding: utf-8 -*-
"""
Created on Wed May  3 21:44:09 2023

@author: rahul
"""

import numpy as np
from scipy.io import savemat,loadmat

nx = 20
nt = 100
mu = 0.05

L = np.linspace(0, 1, nx).reshape(-1, 1)

time = np.linspace(0, 0.5, nt).reshape(-1, 1)

# u0 = np.sin((np.pi)*L)

dx = L[1]-L[0]
dt = time[1]-time[0]
const = (mu/(dx*dx))



DEIM_sensor = loadmat('Burgers_DEIM_phi.mat')['DEIM_sensor'].T
Modes = loadmat('Burgers_DEIM_phi.mat')['Modes']
phi = loadmat('Burgers_DEIM_phi.mat')['phi']
q0 = loadmat('Burgers_DEIM_phi.mat')['q0']
Ar = loadmat('Burgers_DEIM_phi.mat')['Ar']
u_Mat = loadmat("Burgers_FOM.mat")['u_Mat']

q = q0
T = 0
q_Mat = None

def F(nx,q,phi,DEIM_sensor):
    
    u = np.matmul(phi,q)
    F = np.zeros((nx, nx))

    for i in range(1,nx-1):
        if (i == nx-2) :
           F[i,i] = -u[i]
        else:
           F[i,i+1]= u[i]
           F[i,i]=  -u[i]
    a = DEIM_sensor[:,0]
    F_reduced = F[a,:]
    return F_reduced


for t in range(1):
    print("T=",T)
    T = T+dt
    u = np.matmul(phi,q)
    Nonlinear_Term_Reduced = np.matmul(F(nx,q,phi,DEIM_sensor),u)
    Nonlinear_Term = np.matmul(Modes,Nonlinear_Term_Reduced) 
    F_projected_q =  np.matmul(phi.T,Nonlinear_Term)
    q_updated =  q + const*dt*np.matmul(Ar,q) - (dt/dx)*F_projected_q
    
#     if q_Mat is None:
#         q_Mat = q[:,0].reshape((q.shape[0], 1))
#     else:
#         q_Mat = np.append(q_Mat,q[:,0].reshape((q.shape[0], 1)),axis =1) 
    
#     q = q_updated
    
# u_Mat_ROM = np.matmul(phi,q_Mat)
# u_Mat_FOM = u_Mat
# mdic = {"q_Mat": q_Mat}
# savemat("q_Mat_test.mat", mdic)
