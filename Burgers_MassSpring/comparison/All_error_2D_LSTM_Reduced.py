# -*- coding: utf-8 -*-
"""
Created on Thu May 18 22:00:48 2023

@author: rahul
"""

import scipy.io
import matplotlib.pyplot as plt

method = 'LSTMOnlyData'

Data_used = 1

seq_len = 10


# Ground Truth
u_actual = scipy.io.loadmat('../Results.mat')['u_Mat'][:,seq_len+1:]
time_actual = scipy.io.loadmat('../Results.mat')['total_time'].T[:,seq_len+1:]
x_actual = scipy.io.loadmat('../Results.mat')['total_L'][:,seq_len+1:]

# Prediction

u_pred_LSTMOnlyData = scipy.io.loadmat('../Burgers_Dis_{}_{}.mat'.format(method,Data_used))['predicted_output_{}_{}'.format(method,Data_used)].T



method = 'LSTM'

Data_used = 1

seq_len = 10


# Ground Truth
u_actual = scipy.io.loadmat('../Results.mat')['u_Mat'][:,seq_len+1:]
time_actual = scipy.io.loadmat('../Results.mat')['total_time'].T[:,seq_len+1:]
x_actual = scipy.io.loadmat('../Results.mat')['total_L'][:,seq_len+1:]

# Prediction

u_pred_LSTM = scipy.io.loadmat('../Burgers_Dis_{}_{}.mat'.format(method,Data_used))['predicted_output_{}_{}'.format(method,Data_used)].T







fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

#plot Original 

cb = getattr(ax[0], 'contourf')(x_actual,time_actual,abs(u_pred_LSTMOnlyData-u_actual))
fig.colorbar(cb,ax = ax[0])
ax[0].set_title("LSTM")
ax[0].set_xlabel('x',fontsize=24)
ax[0].set_ylabel('t',fontsize=24)
# ax[0].set_xticklabels('x', fontsize=24)

#plot prediciton

cb = getattr(ax[1], 'contourf')(x_actual,time_actual,abs(u_pred_LSTM-u_actual))
fig.colorbar(cb,ax = ax[1])
ax[1].set_title("LSTM-PINN")
ax[1].set_xlabel('x',fontsize=24)
ax[1].set_ylabel('t',fontsize=24)
# ax[1].set_xticklabels('x', fontsize=24)



plt.savefig('error_comparison_2D/error_{}_{}.pdf'.format(method, Data_used), dpi=300)