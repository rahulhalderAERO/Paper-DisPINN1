# -*- coding: utf-8 -*-
"""
Created on Thu May 18 22:00:48 2023

@author: rahul
"""

import scipy.io
import matplotlib.pyplot as plt

Data_list = [1,100]

for Data_used in Data_list:
    
   method = 'ANN'

   Type = 'Data'

   # Ground Truth
   u_actual = scipy.io.loadmat('../Results/Benchmark/Results_Burgers.mat')['u_Mat']
   time_actual = scipy.io.loadmat('../Results/Benchmark/Results_Burgers.mat')['total_time'].T
   x_actual = scipy.io.loadmat('../Results/Benchmark/Results_Burgers.mat')['total_L']

   # Prediction

   u_pred_ANNOnlyData = scipy.io.loadmat('../Results/Burgers/Burgers_Dis_{}_{}_{}.mat'.format(method,Data_used,Type))['predicted_output_{}_{}_{}'.format(method,Data_used,Type)].T
   x_pred = scipy.io.loadmat('../Results/Benchmark/Results_Burgers.mat')['total_time'].T
   time_pred = scipy.io.loadmat('../Results/Benchmark/Results_Burgers.mat')['total_time'].T

   method = 'ANN'

   Type = 'Physics'


   # Ground Truth
   u_actual = scipy.io.loadmat('../Results/Benchmark/Results_Burgers.mat')['u_Mat']
   time_actual = scipy.io.loadmat('../Results/Benchmark/Results_Burgers.mat')['total_time'].T
   x_actual = scipy.io.loadmat('../Results/Benchmark/Results_Burgers.mat')['total_L']

   # Prediction

   u_pred_ANN = scipy.io.loadmat('../Results/Burgers/Burgers_Dis_{}_{}_{}.mat'.format(method,Data_used,Type))['predicted_output_{}_{}_{}'.format(method,Data_used,Type)].T
   x_pred = scipy.io.loadmat('../Results/Benchmark/Results_Burgers.mat')['total_time'].T
   time_pred = scipy.io.loadmat('../Results/Benchmark/Results_Burgers.mat')['total_time'].T


   custom_xticks = [0.2, 0.4, 0.6, 0.8]  # Replace with your desired custom tick values




   fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

   #plot Original 

   cb = getattr(ax[0], 'contourf')(x_actual,time_actual,abs(u_pred_ANNOnlyData-u_actual))
   # fig.colorbar(cb,ax = ax[0])

   colorbar =fig.colorbar(cb,ax = ax[0])
   colorbar.ax.tick_params(labelsize=20)

   ax[0].set_title("ANN",fontsize=24)
   ax[0].set_xlabel('x',fontsize=24)
   ax[0].set_ylabel('t',fontsize=24)
   ax[0].set_xticks(custom_xticks)


   #plot prediciton

   cb = getattr(ax[1], 'contourf')(x_actual,time_actual,abs(u_pred_ANN-u_actual))

   ax[1].set_title("ANN-DisPINN",fontsize=24)


   colorbar =fig.colorbar(cb,ax = ax[1])
   colorbar.ax.tick_params(labelsize=20)


   ax[1].set_xlabel('x',fontsize=24)
   ax[1].set_ylabel('t',fontsize=24)
   ax[1].set_xticks(custom_xticks)


   # Adjust x-axis and y-axis tick label font sizes
   for i in range(2):
    ax[i].tick_params(axis='x', labelsize=20)
    ax[i].tick_params(axis='y', labelsize=20)



   plt.savefig('../Results/Burgers_error_{}_{}.pdf'.format(method, Data_used),bbox_inches = 'tight', dpi=1200, )