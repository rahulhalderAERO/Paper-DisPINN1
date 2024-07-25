import argparse
import torch
from torch.nn import Softplus
from pina import DisPINNANNBurgers_External, Plotter, LabelTensor
from pina.model import FeedForward,LSTM
from problems.burgers_tensor_discrete_ANN_pointwise_Residual1 import Burgers1D

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "-save", action="store_true")
    group.add_argument("-l", "-load", action="store_true")
    parser.add_argument("id_run", help = "number of run", type=int)
    args = parser.parse_args()

    ntotal = 100
    cut_Eqn = 10
    cut_Data = 2
    
    burgers_problem = Burgers1D(ntotal,cut_Eqn,cut_Data)
    
    model = FeedForward(
        layers=[124, 64, 24, 8],
        output_variables=burgers_problem.output_variables,
        input_variables=burgers_problem.input_variables,
        func=Softplus,
    )

    pinn = DisPINNANNBurgers_External(
        burgers_problem,
        model,
        lr=0.006,
        error_norm='mse',
        regularizer=0)

    if args.s:
        pinn.span_tensor_given_pts(
            {'n': 100,'variables': 'all'},
            locations=['A','D','F'])
        pinn.train(6000, 1)
        pinn.save_state('ANN_pina.burger_dis1.{}.{}'.format(args.id_run, args.features))
        
    else:
        pinn.load_state('ANN_pina.burger_dis1.{}.{}'.format(args.id_run, args.features))
        plotter = Plotter()        
        plotter.plot_same_training_test_data(pinn)
        plotter.plot_loss(pinn)

