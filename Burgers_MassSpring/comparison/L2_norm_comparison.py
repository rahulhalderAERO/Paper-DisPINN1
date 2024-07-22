# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 13:39:26 2023

@author: rahul
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

list = [1,3,200,1000]

method = 'ANN'

Benchmark_data = scipy.io.loadmat('Benchmark.mat')['Benchmark']

def L2norm(actual,pred):
    relative_l2_norm = np.linalg.norm(actual - pred) / np.linalg.norm(actual)
    return relative_l2_norm


relative_L2_ANN = np.zeros(len(list))

for i in range(len(list)):
    mat = scipy.io.loadmat('MassSpring_Dis_{}_{}.mat'.format(method,list[i]))
    actual = Benchmark_data[:,2]
    pred = mat['predicted_output_{}_{}'.format(method,list[i])][:,0]
    relative_L2_ANN[i] = L2norm(actual,pred)
    
method = 'LSTM'

relative_L2_LSTM = np.zeros(len(list))

for i in range(len(list)):
    mat = scipy.io.loadmat('MassSpring_Dis_{}_{}.mat'.format(method,list[i]))
    actual = Benchmark_data[11:,2]
    pred = mat['predicted_output_{}_{}'.format(method,list[i])][:,0]
    relative_L2_LSTM[i] = L2norm(actual,pred)


x = np.arange(1,len(list)+1)
plt.figure(figsize=(8, 5))
plt.plot(x, relative_L2_ANN,
         linestyle='--',
         marker='s',
         linewidth=2.5,
         color='red',
         label='ANN-PINN')
plt.plot(x, relative_L2_LSTM,
         linewidth=2.5,
         marker='o',
         color='blue',
         label='LSTM-PINN')
plt.xlabel('Data Points',fontsize=20)
plt.ylabel('Relative L2 norm', fontsize=20)
xtick_locs = np.arange(0, x[-1]+1, 1)
plt.xticks(xtick_locs)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# legend settings
plt.ylim(0.08, 0.23)
plt.xlim(1, 4)

plt.grid(linestyle='dotted')
plt.legend(ncol=2, loc=9, fontsize=20) # 9 means top center
plt.tight_layout()
plt.savefig('Plots_comparison/L2norm_h.pdf', dpi=300)