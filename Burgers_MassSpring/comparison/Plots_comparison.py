# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 15:43:02 2023

@author: rahul
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

list = [1,3,200,1000]

Structure = 'pitch'

h = 1

Benchmark_data = scipy.io.loadmat('Benchmark.mat')['Benchmark']
actual = Benchmark_data[:,h+2]

t = np.arange(0,len(actual)*np.pi/18,np.pi/18)



for i in range(len(list)):
    mat_ANN = scipy.io.loadmat('MassSpring_Dis_{}_{}.mat'.format('ANN',list[i])) 
    mat_LSTM = scipy.io.loadmat('MassSpring_Dis_{}_{}.mat'.format('LSTM',list[i]))  
    pred_ANN = mat_ANN['predicted_output_{}_{}'.format('ANN',list[i])][:,h]
    pred_LSTM = mat_LSTM['predicted_output_{}_{}'.format('LSTM',list[i])][:,h]
    
    plt.figure(figsize=(8, 5))
    plt.plot(t, pred_ANN,
             linewidth=1.5,
             color='red',
             label='ANN-PINN')
    plt.plot(t[11:], pred_LSTM,
             linewidth=1.5,
             color='blue',
             label='LSTM-PINN')
    plt.plot(t, actual,
             linewidth=2.0,
             linestyle='-',
             color='black',
             label='Benchmark')
    plt.xlabel('time (s)',fontsize=20)
    plt.ylabel(r'$\alpha$', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # legend settings
    plt.ylim(-1.0, 1.2)
    plt.xlim(0, 175)
    plt.grid(linestyle='dotted')
    plt.legend(ncol=3, loc=9, fontsize=15) # 9 means top center
    plt.tight_layout()
    plt.savefig('Plots_comparison/{}_{}.pdf'.format(Structure, list[i]), dpi=300)