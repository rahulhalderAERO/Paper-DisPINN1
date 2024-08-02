# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 15:43:02 2023

@author: rahul
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

list = [1,1000]

Structure = 'plunge'

h = 0

Benchmark_data = scipy.io.loadmat('../Results/Benchmark/Results_MassSpring.mat')['Benchmark']
actual = Benchmark_data[:,h+2]

t = np.arange(0,len(actual)*np.pi/18,np.pi/18)



for i in range(len(list)):
    mat_ANN = scipy.io.loadmat('../Results/Mass_Spring/MassSpring_Dis_{}_{}_Physics.mat'.format('ANN',list[i])) 
    mat_AD = scipy.io.loadmat('../Results/Mass_Spring/MassSpring_Dis_{}_{}_Physics.mat'.format('AD',list[i])) 
    mat_LSTM = scipy.io.loadmat('../Results/Mass_Spring/MassSpring_Dis_{}_{}_Physics.mat'.format('LSTM',list[i]))  
    pred_AD = mat_AD['predicted_output_{}_{}_Physics'.format('AD',list[i])][:,h]
    pred_ANN = mat_ANN['predicted_output_{}_{}_Physics'.format('ANN',list[i])][:,h]
    pred_LSTM = mat_LSTM['predicted_output_{}_{}_Physics'.format('LSTM',list[i])][:,h]
    
    plt.figure(figsize=(8, 5))
    
    plt.plot(t, pred_AD,
             linewidth=1.5,
             color='green',
             label='AD-PINN')
    
    plt.plot(t, pred_ANN,
             linewidth=1.5,
             color='red',
             label='ANN-DisPINN')
    plt.plot(t[11:], pred_LSTM,
             linewidth=1.5,
             color='blue',
             label='LSTM-DisPINN')
    plt.plot(t, actual,
             linewidth=2.0,
             linestyle='-',
             color='black',
             label='Benchmark')
    
    
    desired_y_ticks = [-1.5,-0.5, 0.5, 1.5,2.5]
    desired_x_ticks = [0, 50, 100, 150]
    
    
    plt.xlabel(r'$\tau$',fontsize=28)
    plt.ylabel(r'$h$', fontsize=28)
    # plt.xticks(fontsize=20)
    plt.yticks(desired_y_ticks, fontsize=24)    
    plt.xticks(desired_x_ticks, fontsize=24)    

    
    
    # legend settings
    plt.ylim(-1.5,4)
    plt.xlim(-20, 180)
    plt.grid(linestyle='dotted')
    plt.legend(ncol=1, loc='upper left', fontsize=20) # 9 means top center
    plt.tight_layout()
    plt.savefig('../Plot_Results/MassSpring_{}_{}.pdf'.format(Structure,list[i]),bbox_inches = 'tight', dpi=1200)