import sys

sys.path.append('./PINN_LBFGS')
import torch
import wandb
from torch.utils.data import DataLoader
from aux_functions import (FCN, create_dataset, construct_input_vec,
                           plot_results, scan_checkpoint, save_checkpoint,
                           load_checkpoint)
from pathlib import Path
import numpy as np
import click
import os
import matplotlib.pyplot as plt

# Set default dtype to float32
torch.set_default_dtype(torch.float)
# PyTorch random number generator
torch.manual_seed(1234)
# Random number generators in other libraries
np.random.seed(1234)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if device == 'cuda':
    print(torch.cuda.get_device_name())


# %%
@click.command()
@click.option(
    "--data_dir", default='../Data', type=str, help="Directory of training data"
)
@click.option(
    "--checkpoint_dir",
    default="./PINN_checkpoints",
    type=str,
    help="Directory for saving model checkpoints",
)
@click.option(
    "--train_epochs",
    default=1e7,
    type=int,
    help="Number of epochs for which to train PINN (in total)",
)
@click.option(
    "--siren",
    default=True,
    type=bool,
    help="Use sinusoidal activations",
)
def train_PINN(data_dir, checkpoint_dir, train_epochs, siren):
    config = {
        'rir_time': 0.08,
        'check_point_dir': checkpoint_dir,
        'train_epochs': train_epochs,
        'lr': 1e-5,
        'layers': np.array([3, 100, 100, 100, 100, 100, 100, 100, 1]),
        'batch_size': 90,
        'siren': siren,
        'lambda_data': 100.,
        'lambda_pde': 1e-2,
        'lambda_bc': 0.,
        'n_pde_samples': 15000
    }
    data_params = {'data_path': '../Data/ISM_sphere.npz',
                   'n_collocation': 10000,
                   'x_bounds': (-1, 1),
                   'y_bounds': (-1, 1),
                   't_bounds': (0, config['rir_time']),
                   'speed_of_sound': 343.,
                   'batch_size': 10000}
    config = {**config, **data_params}

    wandb.init(config=config, project="PINN_sound_field")

    hparams = wandb.config
    tfnp = lambda x: torch.from_numpy(x).float().to(device)

    # %%
    """Training Data"""
    collocation_data_ref, collocation_data_m, collocation_pde, lookup_reference, lookup_measured = create_dataset(
        data_params)

    collocation_ics = collocation_data_ref[:, np.where(collocation_data_ref[2] == 0)[0]]

    # create test data
    t_test = [0.01, 0.03, 0.05, 0.07, 0.11]  # seconds
    for i, t_ in enumerate(t_test):
        t_test[i] = collocation_data_ref[2, np.argmax(collocation_data_ref[2] > t_)]
    collocation_test = []
    pressure_test = []
    for tt in t_test:
        collocation_test.append(collocation_data_ref[:, np.where(collocation_data_ref[2] == tt)[0]])
        pressure_test.append(lookup_reference[np.where(collocation_data_ref[2] == tt)[0], 0])
    collocation_test = np.array(collocation_test)
    pressure_test = np.array(pressure_test)

    collocation_data_ref, collocation_data_m, collocation_pde, \
    lookup_reference, lookup_measured = (tfnp(collocation_data_ref),
                                         tfnp(collocation_data_m),
                                         tfnp(collocation_pde),
                                         tfnp(lookup_reference),
                                         tfnp(lookup_measured))
    collocation_ics = tfnp(collocation_ics)
    # collocation_test = tfnp(collocation_test)
    # pressure_test = tfnp(pressure_test)

    # %%

    bounds = {'x': hparams.x_bounds,
              'y': hparams.y_bounds,
              't': hparams.t_bounds}

    PINN = FCN(hparams.layers, bounds=bounds, device=device, siren=hparams.siren,
               collocation_data_m=collocation_data_m,
               collocation_pde=collocation_pde,
               collocation_ics=collocation_ics,
               pressure_data=lookup_measured[:,0],
               collocation_test=collocation_test,
               pressure_test=pressure_test,
               lambda_data=hparams.lambda_data, lambda_pde=hparams.lambda_pde, lambda_bc=hparams.lambda_bc)

    wandb.watch(PINN.dnn, PINN.loss_function, log='all')

    '''Optimization'''
    steps = 200000

    optimizer = torch.optim.LBFGS(PINN.dnn.parameters(), lr=.01,
                                  max_iter=steps,
                                  max_eval=None,
                                  tolerance_grad=1e-06,
                                  tolerance_change=1e-10,
                                  history_size=10000,
                                  line_search_fn='strong_wolfe')
    PINN.optimizer = optimizer
    'Neural Network Summary'
    print(PINN.dnn)

    # %%
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if os.path.isdir(checkpoint_dir):
        cp_pinn = scan_checkpoint(checkpoint_dir, "PINN_")

    steps = 0
    if cp_pinn is None:
        last_epoch = -1
    else:
        state_dict_pinn = load_checkpoint(cp_pinn, device)
        PINN.dnn.load_state_dict(state_dict_pinn["net"])
        steps = state_dict_pinn["steps"] + 1
        last_epoch = state_dict_pinn["epoch"]
        optimizer.load_state_dict(state_dict_pinn["optim"])
    itr = 0

    PINN.dnn.train()
    optimizer.step(PINN.closure)


if __name__ == "__main__":
    train_PINN()
# train_PINN()
