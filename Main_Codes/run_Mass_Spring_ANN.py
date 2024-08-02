import argparse
import torch
from torch.nn import Softplus
from pina import DisPINNANNMassSpring, PlotterANNMassSpring, LabelTensor
from pina.model import FeedForward
from problems.Mass_Spring_Dis_with_Data import MassSpring1D
import numpy as np



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "-save", action="store_true")
    group.add_argument("-l", "-load", action="store_true")
    parser.add_argument("id_run", help="number of run", type=int)
    args = parser.parse_args()
    cut_Data_list = ["1",1000]#["1","1a","1b","1c","1d","3","3a","3b","3c","3d","200","200a","200b","200c",1000]
    Type_Run_list = ["Data","Physics"]

    # Define Structural Properties of Mass Spring SystemError
    # For details check Reference [23]
    
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
       model = FeedForward(
        layers=[124, 64, 24, 8],
        output_variables = MassSpring_problem.output_variables,
        input_variables = MassSpring_problem.input_variables,
        func = Softplus,
       ) #

       pinn = DisPINNANNMassSpring(
        MassSpring_problem,
        model,
        lr=0.01,
        error_norm='mse',
        regularizer=0)

       if args.s:
        pinn.span_tensor_given_pts(
            {'n': 1000,'variables': 'all'},
            locations=['D','gamma'])
        pinn.train(10000, 1)
        pinn.save_state('pretrained_model/Mass_Spring/ANN_pina.MassSpring_dis.{}_{}_{}'.format(args.id_run,cut_Data,Type_Run))
       else:
        pinn.load_state('pretrained_model/Mass_Spring/ANN_pina.MassSpring_dis.{}_{}_{}'.format(args.id_run,cut_Data,Type_Run))
        plotter = PlotterANNMassSpring()
        plotter.plot_same_training_test_data(pinn)
        plotter.plot_loss(pinn)

