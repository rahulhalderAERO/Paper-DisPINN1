import argparse
import torch
from torch.nn import Softplus

from pina import DisPINNLSTMBurgers, PlotterLSTM, LabelTensor
from pina.model import LSTM
from problems.burgers_tensor_discrete_LSTM import Burgers1D


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "-save", action="store_true")
    group.add_argument("-l", "-load", action="store_true")
    parser.add_argument("id_run", help = "number of run", type=int)
    args = parser.parse_args()
    cut_Data_list = [1,100]
    Type_Run_list = ["Data","Physics"]
     
    for cut_Data in cut_Data_list:
     for Type_Run in Type_Run_list:
    
       burgers_problem = Burgers1D(cut_Data,Type_Run)
       model = LSTM(input_variables = burgers_problem.input_variables, output_variables = burgers_problem.output_variables,
                 hidden_size = 10, num_layers = 1, seq_length = 10)

       pinn = DisPINNLSTMBurgers(
        burgers_problem,
        model,
        lr=0.1,
        error_norm='mse',
        regularizer=0)

       if args.s:
        pinn.span_tensor_given_pts(
            {'n': 100,'variables': 'all'},
            locations=['D'])
        pinn.train(6000, 1)
        pinn.save_state('pretrained_model/Burgers/LSTM_pina.burger_dis.{}_{}_{}'.format(args.id_run,cut_Data,Type_Run))
        
       else:
        pinn.load_state('pretrained_model/Burgers/LSTM_pina.burger_dis.{}_{}_{}'.format(args.id_run,cut_Data,Type_Run))
        plotter = PlotterLSTM()        
        plotter.plot_same_training_test_data(pinn)
        plotter.plot_loss(pinn)

