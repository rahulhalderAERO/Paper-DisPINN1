# -*- coding: utf-8 -*-
"""
Created on Thu May 18 22:00:48 2023

@author: rahul
"""

import scipy.io
import matplotlib.pyplot as plt


Data_list = [1]

for Data_used in Data_list:

  method = 'ANN'

  Jacobian = 100

  # Ground Truth
  u_actual1 = scipy.io.loadmat('../Results/Benchmark/Results_Burgers.mat')['u_Mat']
  time_actual1 = scipy.io.loadmat('../Results/Benchmark/Results_Burgers.mat')['total_time'].T
  x_actual1 = scipy.io.loadmat('../Results/Benchmark/Results_Burgers.mat')['total_L']

  # Prediction

  u_pred_ANN = scipy.io.loadmat('../Results/Burgers_External/Burgers_Dis_{}External_{}_Physics.mat'.format(method,Data_used))['predicted_output_{}External_{}_Physics'.format(method,Data_used)].T



  method = 'LSTM'


  seq_len = 10

  Jacobian = 100


  # Ground Truth
  u_actual = scipy.io.loadmat('../Results/Benchmark/Results_Burgers.mat')['u_Mat'][:,seq_len+1:]
  time_actual = scipy.io.loadmat('../Results/Benchmark/Results_Burgers.mat')['total_time'].T[:,seq_len+1:]
  x_actual = scipy.io.loadmat('../Results/Benchmark/Results_Burgers.mat')['total_L'][:,seq_len+1:]

  # Prediction

  u_pred_LSTM = scipy.io.loadmat('../Results/Burgers_External/Burgers_Dis_{}External_{}_Physics.mat'.format(method,Data_used))['predicted_output_{}External_{}_Physics'.format(method,Data_used)].T
  custom_xticks = [0.2, 0.4, 0.6, 0.8]  # Replace with your desired custom tick values
  # plt.savefig('../Plot_Results/External_Burgers_error_{}.pdf'.format(Data_used),bbox_inches = 'tight', dpi=1200)






  fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

  #plot Original 

  cb = getattr(ax[0], 'contourf')(x_actual1,time_actual1,abs(u_pred_ANN-u_actual1))

  colorbar = fig.colorbar(cb,ax = ax[0])
  colorbar.ax.tick_params(labelsize=17)

  ax[0].set_title("ANN-DisPINN",fontsize=24)
  ax[0].set_xlabel('x',fontsize=24)
  ax[0].set_ylabel('t',fontsize=24)
  ax[0].set_xticks(custom_xticks)

  # ax[0].set_xticklabels('x', fontsize=24)

  #plot prediciton

  cb = getattr(ax[1], 'contourf')(x_actual,time_actual,abs(u_pred_LSTM-u_actual))

  colorbar = fig.colorbar(cb,ax = ax[1])
  colorbar.ax.tick_params(labelsize=17)


  ax[1].set_title("LSTM-DisPINN",fontsize=24)
  ax[1].set_xlabel('x',fontsize=24)
  ax[1].set_ylabel('t',fontsize=24)
  ax[1].set_xticks(custom_xticks)

  # ax[1].set_xticklabels('x', fontsize=24)


  for i in range(2):
    ax[i].tick_params(axis='x', labelsize=20)
    ax[i].tick_params(axis='y', labelsize=20)



  plt.savefig('../Results/External_Burgers_error_{}.pdf'.format(Data_used),bbox_inches = 'tight', dpi=1200)