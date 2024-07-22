import argparse
import torch
from torch.nn import Softplus
from pina import PINNBurgers, Plotter, LabelTensor
from pina.model import FeedForward
from problems.burgers_with_data import Burgers1D

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "-save", action="store_true")
    group.add_argument("-l", "-load", action="store_true")
    parser.add_argument("id_run", help="number of run", type=int)
    args = parser.parse_args()    
    ntotal = 2000
    cut_Data = 2

    burgers_problem = Burgers1D(ntotal,cut_Data)
    model = FeedForward(
        layers=[30, 20, 10, 5],
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
        pinn.save_state('pina.burger_dis1.{}.{}'.format(args.id_run, args.features))
    else:
        pinn.load_state('pina.burger_dis1.{}.{}'.format(args.id_run, args.features))
        # plotter = Plotter()
        # plotter.plot_same_training_test_data(pinn)
        # plotter.plot_loss(pinn)

