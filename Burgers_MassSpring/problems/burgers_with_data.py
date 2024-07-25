import torch
from pina.problem import TimeDependentProblem, SpatialProblem
from pina.operators import grad
from pina import Condition
from pina.span import Span
import numpy
from pina import LabelTensor
import pandas as pd
from scipy.io import savemat
import os
import random
import numpy as np

output_data = pd.read_csv("U_profile.csv",skiprows = None , header = None)
output_data = output_data.values
output_tensor = (torch.from_numpy(output_data)).float()

input_data = pd.read_csv("xt_loc.csv",skiprows = None , header = None)
input_data = input_data.values
input_tensor = (torch.from_numpy(input_data)).float()

input_tensor = LabelTensor(input_tensor,['x', 't'])
output_tensor = LabelTensor(output_tensor,['u'])

class Burgers1D(TimeDependentProblem, SpatialProblem):

    output_variables = ['u']
    spatial_domain = Span({'x': [0, 1]})
    temporal_domain = Span({'t': [0, 0.5]})
    
    def __init__(self,ntotal,cut_Data):
        self.ntotal = ntotal
        self.cut_Data = cut_Data

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
        du = grad(output_, input_)
        ddu = grad(du, input_, components=['dudx'])
        u_ext = (output_.extract(['u']))
        return (
            du.extract(['dudt']) +
            output_.extract(['u'])*du.extract(['dudx']) -
            (0.05)*ddu.extract(['ddudxdx'])
        )
        
    def nil_dirichlet(self,input_, output_):
        u_expected = 0.0
        return output_.extract(['u']) - u_expected

    def initial_condition(self,input_, output_):
        u_expected = torch.sin(torch.pi*input_.extract(['x']))
        return output_.extract(['u']) - u_expected

    conditions = {
        'gamma1': Condition(Span({'x': 0, 't': [0, 0.5]}), nil_dirichlet),
        'gamma2': Condition(Span({'x':  1, 't': [0,0.5]}), nil_dirichlet),
        't0': Condition(Span({'x': [0, 1], 't': 0}), initial_condition),
        'D': Condition(Span({'x': [0, 1], 't': [0, 0.5]}), burger_equation),
        'E': Condition(input_points = input_tensor,output_points = output_tensor),
    }
