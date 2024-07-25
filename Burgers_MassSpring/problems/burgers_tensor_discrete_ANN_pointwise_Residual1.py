import torch

from pina.problem import TimeDependentProblem, SpatialProblem
from pina.operators import grad
from pina import Condition
from pina.span import Span
import numpy
from problems.Burgers_DiscreteTorch_Class_pointwise_Actual import Burgers_Discrete
from scipy.io import savemat
import os
from pina import LabelTensor
import pandas as pd
import random
import numpy as np
import sys


output_data = pd.read_csv("Utot_profile1.csv",skiprows = None , header = None)
output_data = output_data.values
output_tensor = (torch.from_numpy(output_data)).float()

input_data = pd.read_csv("Time_profile.csv", skiprows = None , header = None)
input_data = input_data.values
input_tensor = (torch.from_numpy(input_data)).float()

list_u = []
no_list = 20
for i in range(no_list):
    list_u.append('u_{}'.format(i))

input_tensor = LabelTensor(input_tensor,['t'])
output_tensor = LabelTensor(output_tensor,list_u)


class Burgers1D(TimeDependentProblem):
    
    list_u = []
    no_list = 20
    for i in range(no_list):
        list_u.append('u_{}'.format(i))
    output_variables = list_u
    temporal_domain = Span({'t': input_tensor.reshape(-1,1)})
    
    def __init__(self,ntotal,cut_Eq,cut_Data):
        self.ntotal = ntotal
        self.cut_Eq = cut_Eq
        self.cut_Data = cut_Data

    def rand_choice_integer_Eq(self):
        
        list1= [0,1,2,3,4]
        
        list2=[]
        for i in range(self.cut_Eq):
            r=random.randint(5,self.ntotal-1)
            if r not in list1: list1.append(r)
        for i in list1:
            list2.append(i)
        return np.array(list2)
      
    def rand_choice_integer_Data(self):
        
        list1= [0]
        list2=[]
        for i in range(self.cut_Data):
            r=random.randint(1,self.ntotal-1)
            if r not in list1: list1.append(r)
        for i in list1:
            list2.append(i)
        return np.array(list1)
    
    def burger_equation(self,input_, output_):

            
        nx = 20
        nt = 100
        mu = 0.05
        L = torch.linspace(0, 1, nx).reshape(-1, 1)
        
        # ===== Add This Part for Modified Simulation ===========================================================================================================
        
        for i in range(2,output_.size(0)-2):                      
          for j in range(1,output_.size(1)-1):                        
            
            # ===== Residual Computation ==========================================================================

            
            # print("The size of the output_.size(1) is ====", output_.size(1))
            Rresidual = Burgers_Discrete(L, input_.extract(['t']), j, mu, output_[i,j], output_[i,j+1], output_[i,j-1], output_[i+1,j]).Burgers_Residual()
            
            if (i == 2 and j == 1 ): 
                
                new_tensor = (Rresidual)
                                
            else:
                
                Residual_Dis = (Rresidual)
                new_tensor = torch.cat((new_tensor,Residual_Dis), dim = 0)
   


        new_tensor = new_tensor.reshape(-1,1)
        
        mydictionary = {'new_tensor':new_tensor}
        
        
        return mydictionary
    
    def burger_equation_derivative(self,input_, output_):
        nx = 20
        nt = 100
        mu = 0.05
        L = torch.linspace(0, 1, nx).reshape(-1, 1)
        
        # ===== Add This Part for Modified Simulation ===========================================================================================================
        
        Tot_Sim = 0 
        
        for i in range(2,output_.size(0)-2):                      
          for j in range(1,output_.size(1)-1):
              Tot_Sim = Tot_Sim + 1                        

        
        # new_tensor = torch.zeros(Tot_Sim)
        dR_dU_Dis = torch.zeros(Tot_Sim,output_.size(0)*output_.size(1))


        k = 0        
        for i in range(2,output_.size(0)-2):                      
          for j in range(1,output_.size(1)-1):                        
                        
            # ===== dR_DU Computation Discrete ================================================================

            
            dR_dU_p = torch.zeros(1,4)
            dR_dU_m = torch.zeros(1,4)
            dR_dU_p[0,0] = Burgers_Discrete(L, input_.extract(['t']), j, mu, output_[i,j], output_[i,j+1], output_[i,j-1]+0.00001, output_[i+1,j]).Burgers_Residual()
            dR_dU_p[0,1] = Burgers_Discrete(L, input_.extract(['t']), j, mu, output_[i,j]+0.00001, output_[i,j+1], output_[i,j-1], output_[i+1,j]).Burgers_Residual()
            dR_dU_p[0,2] = Burgers_Discrete(L, input_.extract(['t']), j, mu, output_[i,j], output_[i,j+1]+0.00001, output_[i,j-1], output_[i+1,j]).Burgers_Residual()
            dR_dU_p[0,3] = Burgers_Discrete(L, input_.extract(['t']), j, mu, output_[i,j], output_[i,j+1], output_[i,j-1], output_[i+1,j]+0.00001).Burgers_Residual()
            
            
            dR_dU_m[0,0] = Burgers_Discrete(L, input_.extract(['t']), j, mu, output_[i,j], output_[i,j+1], output_[i,j-1]-0.00001, output_[i+1,j]).Burgers_Residual()
            dR_dU_m[0,1] = Burgers_Discrete(L, input_.extract(['t']), j, mu, output_[i,j]-0.00001, output_[i,j+1], output_[i,j-1], output_[i+1,j]).Burgers_Residual()
            dR_dU_m[0,2] = Burgers_Discrete(L, input_.extract(['t']), j, mu, output_[i,j], output_[i,j+1]-0.00001, output_[i,j-1], output_[i+1,j]).Burgers_Residual()
            dR_dU_m[0,3] = Burgers_Discrete(L, input_.extract(['t']), j, mu, output_[i,j], output_[i,j+1], output_[i,j-1], output_[i+1,j]-0.00001).Burgers_Residual()
            dR_dU = (dR_dU_p - dR_dU_m)/(2*0.00001)  
            
            dR_dU_Dis[k,i*output_.size(1)+j-1:i*output_.size(1)+j-1+3] = dR_dU[0,0:3]
            dR_dU_Dis[k,(i+1)*output_.size(1)+j] = dR_dU[0,3]
            k = k+1     
        
        mydictionary = {'Jacobian_Dis':dR_dU_Dis}
        
        
        return mydictionary
    
    def nil_dirichlet_0(self,input_, output_):
        value = 0.0
        return (output_.extract(['u_0']) - value)
    
    def nil_dirichlet_L(self,input_, output_):
        value = 0.0
        return (output_.extract(['u_19']) - value)
    
    def initial_condition(self,input_, output_):
        nx = 20
        L = torch.linspace(0, 1, nx).reshape(-1, 1)
        u_expected = torch.sin(torch.pi*L)
        return output_[0,:] - u_expected
    
    conditions = {
        'A': Condition(Span({'t': input_tensor.reshape(-1,1)}), burger_equation_derivative),
        'D': Condition(Span({'t': input_tensor.reshape(-1,1)}), burger_equation),
        'F': Condition(Span({'t': input_tensor.reshape(-1,1)}), [nil_dirichlet_0,nil_dirichlet_L]),
        'E': Condition(input_points = input_tensor,output_points = output_tensor),
    }
