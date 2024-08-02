import torch
import numpy as np
from pina.problem import TimeDependentProblem
from pina.operators import grad
from pina import Condition
from pina.span import Span
import numpy
from scipy.io import savemat
import os
import pandas as pd
from pina.label_tensor import LabelTensor
import random




output_data = pd.read_csv("Input_Data/ha_output.csv",skiprows = None , header = None)
output_data = output_data.values
output_tensor = torch.from_numpy(output_data)

input_data = pd.read_csv("Input_Data/t_input.csv",skiprows = None , header = None)
input_data = 0.01*input_data.values
input_tensor = torch.from_numpy(input_data)

input_tensor = LabelTensor(input_tensor,['t'])
output_tensor = LabelTensor(output_tensor[:,2:4],['h', 'a'])


print(input_data.shape[0])




class MassSpring1D(TimeDependentProblem):
    
    output_variables = ['h','a']
    temporal_domain = Span({'t': [0, 999*numpy.pi/1800]})

    def __init__(self,M,K,vf,nt,cut_Data,Type_Run):
        self.M = M
        self.K = K
        self.vf = vf
        self.const = (self.vf*self.vf)/(2*(torch.pi)); 
        self.nt = nt
        self.cut_Data = cut_Data
        self.Type_Run = Type_Run
        
    
    def getM(self):
        return self.M
    
    def getK(self):
        return self.K  
    
    def rand_choice_integer_Data(self):        
        return self.cut_Data
        
    def runtype(self):        
        return self.Type_Run    
    
    def Eqn_F1(self,input_, output_): 
        cut_Residual = 5    
        dX = grad(output_, input_)
        ddh = grad(dX, input_, components=['dhdt'])
        dda = grad(dX, input_, components=['dadt']) 
        GE = (1/10000)*(self.getM[0,0]*ddh.extract(['ddhdtdt']) + self.getM[0,1]*dda.extract(['ddadtdt'])) + self.getK[0,0]*output_.extract(['h']) + (torch.sin(10*input_.extract(['t'])))
        return GE[cut_Residual:]
        
        
        
        
    def Eqn_F2(self, input_, output_):
        cut_Residual = 5    
        dX = grad(output_, input_)
        ddh = grad(dX, input_, components=['dhdt'])
        dda = grad(dX, input_, components=['dadt'])
        GE = (1/10000)*(self.getM[1,0]*ddh.extract(['ddhdtdt']) + self.getM[1,1]*dda.extract(['ddadtdt'])) + self.getK[1,1]*output_.extract(['a']) + 2*(torch.sin(20*input_.extract(['t'])))
        return GE[cut_Residual:]
        
    def nil_dirichlet_h(self,input_, output_):
        value = 0.0
        return (output_.extract(['h']) - value)
    
    def nil_dirichlet_a(self,input_, output_):
        value = 0.0
        return (output_.extract(['a']) - value)

    conditions = {
        'D': Condition(Span({'t': [0, 999*numpy.pi/1800]}), [Eqn_F1,Eqn_F2]),
        'E': Condition(input_points = input_tensor,output_points = output_tensor),
    }
