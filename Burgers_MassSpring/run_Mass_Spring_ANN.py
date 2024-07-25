import argparse
import torch
from torch.nn import Softplus
from pina import DisPINNANNMassSpring, Plotter, LabelTensor
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
    
    MassSpring_problem = MassSpring1D(M_2dof,K_2dof,vf,nt)
    
    
    model = FeedForward(
        layers=[124, 64, 24, 8],
        output_variables = MassSpring_problem.output_variables,
        input_variables = MassSpring_problem.input_variables,
        func = Softplus,
    )

    pinn = DisPINNANNMassSpring(
        MassSpring_problem,
        model,
        lr=0.01,
        error_norm='mse',
        regularizer=0.00001)

    if args.s:
        pinn.span_tensor_given_pts(
            {'n': 1000,'variables': 'all'},
            locations=['D','gamma'])
        pinn.train(25000, 1)
        pinn.save_state('pina.MassSpring.{}.{}'.format(args.id_run, args.features))
    else:
        pinn.load_state('pina.MassSpring.{}.{}'.format(args.id_run, args.features))
        plotter = Plotter()
        plotter.plot(pinn)
        plotter.plot_loss(pinn)

