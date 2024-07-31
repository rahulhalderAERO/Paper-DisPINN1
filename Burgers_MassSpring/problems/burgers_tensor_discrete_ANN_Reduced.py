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
    
    def __init__(self,cut_Data,Type_Run,DEIM_sensor,Modes,phi,Ar):
    
        self.cut_Data = cut_Data
        self.Type_Run = Type_Run
        self.DEIM_sensor = DEIM_sensor
        self.Modes = Modes
        self.phi = phi
        self.Ar = Ar
      
    def rand_choice_integer_Data(self):        
        return self.cut_Data
        
    def runtype(self):        
        return self.Type_Run
    
    def burger_equation(self,input_, output_):
    
        nx = 20
        nt = 100
        mu = 0.05
        L = torch.linspace(0, 1, nx).reshape(-1, 1)
        for i in range(2,output_.size(0)-2):
            Burgers_Prob = Burgers_Discrete(L,input_.extract(['t']), mu, output_[i,:],output_[i+1,:],self.DEIM_sensor,self.Modes,self.phi,self.Ar)
            if (i == 2):
                new_tensor = (Burgers_Prob.Burgers_Residual())                
            else:
                Residual_Dis = (Burgers_Prob.Burgers_Residual())
                new_tensor = torch.cat((new_tensor,Residual_Dis), dim = 0)               
        return new_tensor
    
    conditions = {
        'D': Condition(Span({'t': input_tensor.reshape(-1,1)}), [burger_equation]),
        'E': Condition(input_points = input_tensor,output_points = output_tensor),
    }
