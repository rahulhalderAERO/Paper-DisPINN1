import argparse
import torch
from torch.nn import Softplus
from pina import DisPINNANNBurgers_Reduced, Plotter, LabelTensor
from pina.model import FeedForward,LSTM
from scipy.io import loadmat
from problems.burgers_tensor_discrete_ANN_Reduced import Burgers1D

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "-save", action="store_true")
    group.add_argument("-l", "-load", action="store_true")
    parser.add_argument("id_run", help = "number of run", type=int)
    args = parser.parse_args()

    ntotal = 100
    cut_Eqn = 10
    cut_Data = 1
    
    DEIM_sensor = loadmat('Burgers_DEIM_phi.mat')['DEIM_sensor'].T
    DEIM_sensor = torch.from_numpy(DEIM_sensor).float()
    Modes = loadmat('Burgers_DEIM_phi.mat')['Modes']
    Modes = torch.from_numpy(Modes).float()
    phi = loadmat('Burgers_DEIM_phi.mat')['phi']
    phi = torch.from_numpy(phi).float()
    Ar = loadmat('Burgers_DEIM_phi.mat')['Ar']
    Ar = torch.from_numpy(Ar).float()
    
    
    burgers_problem = Burgers1D(ntotal,cut_Eqn,cut_Data,DEIM_sensor,Modes,phi,Ar)
    
    model = FeedForward(
        layers = [124, 64, 24, 8],
        output_variables=burgers_problem.output_variables,
        input_variables=burgers_problem.input_variables,
        func=Softplus,
    )

    pinn = DisPINNANNBurgers_Reduced(
        burgers_problem,
        model,
        lr=0.006,
        error_norm='mse',
        regularizer=0)

    if args.s:
        pinn.span_tensor_given_pts(
            {'n': 100,'variables': 'all'},
            locations=['D'])
        pinn.train(6000, 1)
        pinn.save_state('pina.burger_dis1.{}.{}'.format(args.id_run, args.features))
        plotter = Plotter()        
        plotter.plot_same_training_test_data(pinn)
        plotter.plot_loss(pinn)
    # else:
        # pinn.load_state('pina.burger_dis1.{}.{}'.format(args.id_run, args.features))
        # plotter = Plotter()
        # plotter.plot(pinn)
        # plotter.plot_loss(pinn)

