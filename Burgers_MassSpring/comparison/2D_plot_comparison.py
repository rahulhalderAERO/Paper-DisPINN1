# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 15:43:02 2023

@author: rahul
"""

import scipy.io
import matplotlib.pyplot as plt

method = 'ANNOnlyData'

Data_used = 0


# Ground Truth
u_actual = scipy.io.loadmat('../Results.mat')['u_Mat']
time_actual = scipy.io.loadmat('../Results.mat')['total_time'].T
x_actual = scipy.io.loadmat('../Results.mat')['total_L']

# Prediction

u_pred = scipy.io.loadmat('../Burgers_Dis_{}_{}.mat'.format(method,Data_used))['predicted_output_{}_{}'.format(method,Data_used)].T
x_pred = scipy.io.loadmat('../Results.mat')['total_time'].T
time_pred = scipy.io.loadmat('../Results.mat')['total_time'].T




fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))

#plot Original 

cb = getattr(ax[0], 'contourf')(x_actual,time_actual,u_actual)
fig.colorbar(cb,ax = ax[0])
ax[0].set_title("Ground Truth")
ax[0].set_xlabel('x',fontsize=18)
ax[0].set_ylabel('t',fontsize=18)

#plot prediciton

cb = getattr(ax[1], 'contourf')(x_actual,time_actual,u_pred)
fig.colorbar(cb,ax = ax[1])
ax[1].set_title("Prediction")
ax[1].set_xlabel('x',fontsize=18)
ax[1].set_ylabel('t',fontsize=18)

#plot absolute error

cb = getattr(ax[2], 'contourf')(x_actual,time_actual,abs(u_pred-u_actual))
fig.colorbar(cb,ax = ax[2])
ax[2].set_title("error (Absolute Error)")
ax[2].set_xlabel('x',fontsize=18)
ax[2].set_ylabel('t',fontsize=18)

# plt.savefig('Plots_comparison_2D/{}_{}.pdf'.format(method, Data_used), dpi=300)

