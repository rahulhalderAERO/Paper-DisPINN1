# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 14:43:09 2023

@author: rahul
"""

import numpy as np


class DEIM():
    
    def __init__(self, Data, DEIM_Pts_No):
        self.Data = Data 
        self.DEIM_Pts_No = DEIM_Pts_No
        
    
    
    
    def deim(self):
                
        def max_index(input_list):
            max = input_list[0]
            index = 0
            for i in range(1,len(input_list)):
             if input_list[i] > max:
                max = input_list[i]
                index = i
            return index
        
        
        u, s, vh = np.linalg.svd(self.Data, full_matrices=False)
        u_trancated = u[:,0:self.DEIM_Pts_No]
        
        # Find out the DEIM control points and the DEIM modes
        
        DEIM_sensor = np.zeros(self.DEIM_Pts_No)
        Ui = u_trancated[:,0]
        index_DEIM = max_index(abs(Ui))    
        DEIM_sensor[0] = index_DEIM
        U_DEIM = Ui.reshape(-1,1)
        
        for i in range(1,self.DEIM_Pts_No):
            
            
              if (i == 1):
                  nonzeroind = np.nonzero(DEIM_sensor)[0].reshape(-1,1)
                  A_PSI = U_DEIM[int(DEIM_sensor[nonzeroind]),:]
                  B_PSI = u_trancated[int(DEIM_sensor[nonzeroind]),i]
                  c = B_PSI/A_PSI
                  r = abs(u_trancated[:,i].reshape(-1,1) - U_DEIM*c);
                  index_DEIM = max_index(list(r))
                  DEIM_sensor[i]=index_DEIM
                  U_DEIM = u_trancated[:,0:i+1]
              else:
                  vector = np.vectorize(np.int_)
                  nonzeroind = np.nonzero(DEIM_sensor)[0]
                  A_PSI = (U_DEIM[vector(DEIM_sensor[nonzeroind]),:])
                  B_PSI = (u_trancated[vector(DEIM_sensor[nonzeroind]),i]).reshape(-1,1)
                  A_inv = np.linalg.pinv(A_PSI)
                  c = np.matmul(A_inv,B_PSI)
                  r = abs(u_trancated[:,i].reshape(-1,1) - np.matmul(U_DEIM,c));
                  index_DEIM = max_index(list(r))
                  DEIM_sensor[i] = (index_DEIM)
                  U_DEIM = u_trancated[:,0:i+1]

        DEIM_Dict = {'DEIM_sensor':DEIM_sensor.astype(int),'U_DEIM':U_DEIM}
        
        
        return DEIM_Dict