import argparse
import torch
from torch.nn import Softplus

from pina import DisPINNLSTMBurgers_External, Plotter, LabelTensor
from pina.model import FeedForward,LSTM
from problems.burgers_tensor_discrete_LSTM_pointwise_Residual1 import Burgers1D


class myFeature(torch.nn.Module):
    """
    Feature: sin(pi*x)
    """
    def __init__(self, idx):
        super(myFeature, self).__init__()
        self.idx = idx

    def forward(self, x):
        return LabelTensor(torch.sin(torch.pi * x.extract(['x'])), ['sin(x)'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "-save", action="store_true")
    group.add_argument("-l", "-load", action="store_true")
    parser.add_argument("id_run", help = "number of run", type=int)
    parser.add_argument("features", help = "extra features", type=int)
    args = parser.parse_args()

    feat = [myFeature(0)] if args.features else []
    ntotal = 100
    cut_Eqn = 10
    cut_Data = 1
    
    burgers_problem = Burgers1D(ntotal,cut_Eqn,cut_Data)
    model = LSTM(input_variables = burgers_problem.input_variables, output_variables = burgers_problem.output_variables,
                 hidden_size = 10, num_layers = 1, seq_length = 10)

    pinn = DisPINNLSTMBurgers_External(
        burgers_problem,
        model,
        lr=0.1,
        error_norm='mse',
        regularizer=0)

    if args.s:
        pinn.span_tensor_given_pts(
            {'n': 100,'variables': 'all'},
            locations=['A','D','F'])
        pinn.train(6000, 1)
        pinn.save_state('pina.burger_dis_LSTM.{}.{}'.format(args.id_run, args.features))
        # plotter = Plotter()        
        # plotter.plot_same_training_test_data(pinn)
        # plotter.plot_loss(pinn)
    else:
        pinn.load_state('pina.burger_dis_LSTM.{}.{}'.format(args.id_run, args.features))
        plotter = Plotter()
        plotter.plot_same_training_test_data(pinn)
        plotter.plot_loss(pinn)

