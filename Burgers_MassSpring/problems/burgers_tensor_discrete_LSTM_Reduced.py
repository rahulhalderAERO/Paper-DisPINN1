import torch

from pina.problem import TimeDependentProblem, SpatialProblem
from pina.operators import grad
from pina import Condition
from pina.span import Span
import numpy
from problems.Burgers_DiscreteTorch_Class_Reduced_II import Burgers_Discrete
from scipy.io import savemat
import os
from pina import LabelTensor
import pandas as pd
import random
import numpy as np

output_data = pd.read_csv("qtot_profile.csv",skiprows = None , header = None)
output_data = output_data.values
output_tensor = (torch.from_numpy(output_data)).float()

input_data = pd.read_csv("Time_profile.csv",skiprows = None , header = None)
input_data = input_data.values
input_tensor = (torch.from_numpy(input_data)).float()

list_q = []
no_list = 10
for i in range(no_list):
    list_q.append('q_{}'.format(i))

input_tensor = LabelTensor(input_tensor,['t'])
output_tensor = LabelTensor(output_tensor,list_q)


class Burgers1D(TimeDependentProblem):
    
    list_q = []
    no_list = 10
    for i in range(no_list):
        list_q.append('q_{}'.format(i))
    output_variables = list_q
    temporal_domain = Span({'t': input_tensor.reshape(-1,1)})
    
    def __init__(self,ntotal,cut_Eq,cut_Data,DEIM_sensor,Modes,phi,Ar):
        self.ntotal = ntotal
        self.cut_Eq = cut_Eq
        self.cut_Data = cut_Data
        self.DEIM_sensor = DEIM_sensor
        self.Modes = Modes
        self.phi = phi
        self.Ar = Ar

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
            r=random.randint(1,self.ntotal-20)
            if r not in list1: list1.append(r)
        for i in list1:
            list2.append(i)
        return np.array(list2)
    
    def burger_equation(self,input_, output_):
        
        def delete(arr, ind, dim):
            skip = [i for i in range(arr.size(dim)) if i != ind]
            indices = [slice(None) if i != dim else skip for i in range(arr.ndim)]
            return arr[indices]
            
        nx = 20
        nt = 100
        mu = 0.05
        L = torch.linspace(0, 1, nx).reshape(-1, 1)
        list_array = self.rand_choice_integer_Eq 
        for i in range(10,output_.size(0)-5):
            Burgers_Prob = Burgers_Discrete(L,input_.extract(['t']), mu, output_[i,:],output_[i+1,:],self.DEIM_sensor,self.Modes,self.phi,self.Ar)
            if (i == 10):
                new_tensor = (Burgers_Prob.Burgers_Residual())                
                # for k in range(len(list_array)):
                    # val_row = list_array[k]
                    # new_tensor = delete(new_tensor,val_row-k,0)
            else:
                Residual_Dis = (Burgers_Prob.Burgers_Residual())
                # for k in range(len(list_array)):
                    # val_row = list_array[k]
                    # Residual_Dis = delete(Residual_Dis,val_row-k,0)
                new_tensor = torch.cat((new_tensor,Residual_Dis), dim = 0) 
        
        return new_tensor
        

    conditions = {
        'D': Condition(Span({'t': input_tensor.reshape(-1,1)}), [burger_equation]),
        'E': Condition(input_points = input_tensor,output_points = output_tensor),
    }
