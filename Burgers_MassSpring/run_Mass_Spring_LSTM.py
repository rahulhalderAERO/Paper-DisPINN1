import argparse
import torch
from torch.nn import Softplus
from pina import DisPINNLSTMMassSpring, PlotterLSTMMassSpring, LabelTensor
from pina.model import FeedForward,LSTM
from problems.Mass_Spring_Dis_with_Data import MassSpring1D
import numpy as np


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "-save", action="store_true")
    group.add_argument("-l", "-load", action="store_true")
    parser.add_argument("id_run", help="number of run", type=int)
    args = parser.parse_args()
    cut_Data_list = ["1","1a","1b","1c","1d","3","3a","3b","3c","3d","200","200a","200b","200c",1000]
    Type_Run_list = ["Data","Physics"]
    # Define Structural Properties
    
    M_2dof = np.zeros((2,2))
    K_2dof = np.zeros((2,2))
    M_2dof[0,0] = 1
    M_2dof[0,1] = 1.8 
    M_2dof[1,0] = 1.8
    M_2dof[1,1] = 3.48
    K_2dof[0,0] = 1.0
    K_2dof[0,1] = 0.0
    K_2dof[1,0] = 0.0
    K_2dof[1,1] = 3.48
    vf  = 0.8410
    nt = 1000
    
    for cut_Data in cut_Data_list:
     for Type_Run in Type_Run_list:
     
       MassSpring_problem = MassSpring1D(M_2dof,K_2dof,vf,nt,cut_Data,Type_Run)    
       model = LSTM(input_variables = MassSpring_problem.input_variables, output_variables = MassSpring_problem.output_variables,
                 hidden_size = 10, num_layers = 1, seq_length = 10)

       pinn = DisPINNLSTMMassSpring(
        MassSpring_problem,
        model,
        lr = 0.006,
        error_norm='mse',
        regularizer=0)

       if args.s:
        pinn.span_tensor_given_pts(
            {'n': 1000,'variables': 'all'},
            locations=['D','gamma'])
        pinn.train(5000, 1)        
        pinn.save_state('meta_data/Mass_Spring/LSTM_pina.MassSpring_dis.{}_{}_{}'.format(args.id_run,cut_Data,Type_Run))
       else:
        pinn.load_state('meta_data/Mass_Spring/LSTM_pina.MassSpring_dis.{}_{}_{}'.format(args.id_run,cut_Data,Type_Run))
        plotter = PlotterLSTMMassSpring()        
        plotter.plot_same_training_test_data(pinn)
        plotter.plot_loss(pinn)

