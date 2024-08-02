# -*- coding: utf-8 -*-
"""
Created on Thu May 18 22:00:48 2023

@author: rahul
"""

import scipy.io
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np



method = 'LSTM'
phi =loadmat('../Input_Data/Burgers_DEIM_phi.mat')['phi']


Data_used = 1

seq_len = 10


# Ground Truth
u_actual = scipy.io.loadmat('../Results/Benchmark/Results_Burgers.mat')['u_Mat'][:,seq_len+1:]
time_actual = scipy.io.loadmat('../Results/Benchmark/Results_Burgers.mat')['total_time'].T[:,seq_len+1:]
x_actual = scipy.io.loadmat('../Results/Benchmark/Results_Burgers.mat')['total_L'][:,seq_len+1:]


# Prediction

u_red_1 = scipy.io.loadmat('../Results/Burgers_Reduced/Burgers_Dis_{}Reduced_{}_Physics.mat'.format(method,Data_used))['predicted_output_{}Reduced_{}_Physics'.format(method,Data_used)].T
u_full_1 = np.matmul(phi,u_red_1)



method = 'LSTM'

Data_used = 100

seq_len = 10


# Ground Truth
u_actual = scipy.io.loadmat('../Results/Benchmark/Results_Burgers.mat')['u_Mat'][:,seq_len+1:]
time_actual = scipy.io.loadmat('../Results/Benchmark/Results_Burgers.mat')['total_time'].T[:,seq_len+1:]
x_actual = scipy.io.loadmat('../Results/Benchmark/Results_Burgers.mat')['total_L'][:,seq_len+1:]

# Prediction

u_red_100 = scipy.io.loadmat('../Results/Burgers_Reduced/Burgers_Dis_{}Reduced_{}_Physics.mat'.format(method,Data_used))['predicted_output_{}Reduced_{}_Physics'.format(method,Data_used)].T
u_full_100 = np.matmul(phi,u_red_100)


custom_xticks = [0.2, 0.4, 0.6, 0.8]  # Replace with your desired custom tick values





fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

#plot Original 

cb = getattr(ax[0], 'contourf')(x_actual,time_actual,abs(u_full_1-u_actual))

colorbar =fig.colorbar(cb,ax = ax[0])
colorbar.ax.tick_params(labelsize=20)

ax[0].set_title("Data used = 1",fontsize=24)
ax[0].set_xlabel('x',fontsize=24)
ax[0].set_ylabel('t',fontsize=24)
ax[0].set_xticks(custom_xticks)

#plot prediciton

cb = getattr(ax[1], 'contourf')(x_actual,time_actual,abs(u_full_100-u_actual))

colorbar =fig.colorbar(cb,ax = ax[1])
colorbar.ax.tick_params(labelsize=20)


ax[1].set_title("Data used = 100",fontsize=24)
ax[1].set_xlabel('x',fontsize=24)
ax[1].set_ylabel('t',fontsize=24)
ax[1].set_xticks(custom_xticks)

# Adjust x-axis and y-axis tick label font sizes
for i in range(2):
    ax[i].tick_params(axis='x', labelsize=20)
    ax[i].tick_params(axis='y', labelsize=20)




plt.savefig('../Results/BurgersReduced_error_{}_{}.pdf'.format(method, Data_used),bbox_inches = 'tight', dpi=1200, )