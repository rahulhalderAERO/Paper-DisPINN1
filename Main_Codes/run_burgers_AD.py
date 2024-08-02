import argparse
import torch
from torch.nn import Softplus
from pina import PINNBurgers, PlotterAD, LabelTensor
from pina.model import FeedForward
from problems.burgers_with_data import Burgers1D

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "-save", action="store_true")
    group.add_argument("-l", "-load", action="store_true")
    parser.add_argument("id_run", help="number of run", type=int)
    args = parser.parse_args()    
    cut_Data_list = [1,100]
    Type_Run_list = ["Physics"]
    
    
    for cut_Data in cut_Data_list:
     for Type_Run in Type_Run_list:

       burgers_problem = Burgers1D(cut_Data,Type_Run)
       model = FeedForward(
        layers=[124, 64, 24, 8],
        output_variables=burgers_problem.output_variables,
        input_variables=burgers_problem.input_variables,
        func=Softplus,
       )

       pinn = PINNBurgers(
        burgers_problem,
        model,
        lr = 0.006,
        error_norm='mse',
        regularizer=0)

       if args.s:
        pinn.span_pts(
            {'n': 100, 'mode': 'grid', 'variables': 't'},
            {'n': 20, 'mode': 'grid', 'variables': 'x'},
            locations=['D','gamma1', 'gamma2', 't0'])
        pinn.train(6000, 10)
        pinn.save_state('pretrained_model/Burgers/AD_pina.burger_dis.{}_{}_{}'.format(args.id_run,cut_Data,Type_Run))
       else:
        pinn.load_state('pretrained_model/Burgers/AD_pina.burger_dis.{}_{}_{}'.format(args.id_run,cut_Data,Type_Run))
        plotter = PlotterAD()
        plotter.plot_same_training_test_data(pinn)
        plotter.plot_loss(pinn)

