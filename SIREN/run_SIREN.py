import torch
import wandb
from torch.utils.data import DataLoader
from aux_functions_siren import (FCN, GreensFunAnalytical, construct_input_vec,
                                 plot_results, scan_checkpoint, save_checkpoint,
                                 load_checkpoint, WaveSource)
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
    default=1e6,
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
        'layers': np.array([3, 100, 100, 100, 100, 100, 100, 100, 100, 100, 1]),  # 9 hidden layers
        'batch_size': 90,
        'siren': siren,
        'lambda_data': 1.,
        'lambda_pde': 1e-3,
        'lambda_bc': 0.,
        'n_pde_samples': 30000
    }
    wandb.init(config=config, project="PINN_sound_field")

    hparams = wandb.config
    p = Path(data_dir + '/ISM_sphere.npz')
    if not p.exists():
        p = Path('./Data/ISM_sphere.npz')
    data = np.load(p.resolve())
    keys = [key for key in data.keys()]
    print("datafile keys: ", keys)
    rirdata = data['reference_data']
    fs = data['fs']
    grid = data['grid_reference']

    # %%
    """Training Data"""
    data = rirdata[:, :int(hparams.rir_time * fs)]  # truncate
    rirdata = rirdata / np.max(abs(rirdata))
    data = data / np.max(abs(data))
    # t = np.linspace(0, hparams.rir_time, int(hparams.rir_time * fs))  # Does not need normalisation if < 1 second
    t_ind = np.arange(0, int(hparams.rir_time * fs))
    t = np.linspace(0., rirdata.shape[-1] / fs, rirdata.shape[-1])  # Does not need normalisation if < 1 second
    x_true = grid[0]
    y_true = grid[1]
    # boundary indices
    boundary_ind = np.argwhere(x_true ** 2 + y_true ** 2 > .9 ** 2)
    # regular point indices
    mask = np.ones(grid.shape[1], dtype=bool)
    mask[boundary_ind] = False
    reg_ind = np.argwhere(mask == True)
    # random interpolation indices
    interp_indx = np.random.choice(reg_ind.squeeze(-1), 20, replace=False)  # Randomly chosen points for Interior
    # add to boundary indices
    data_ind = np.vstack((boundary_ind, interp_indx[..., None])).squeeze(-1)
    # data_ind = np.random.choice(reg_ind.squeeze(-1), 850, replace=False)
    # data_ind = np.arange(0, len(x_true))
    # regular point grid
    mask = np.ones(grid.shape[1], dtype=bool)
    mask[data_ind] = False
    # grid_rest = grid[:, mask]
    # %%

    # dataset = PINNDataset(data, x_true, y_true, t, data_ind, boundary_ind,
    #                       t_ind, n_pde_samples=hparams.n_pde_samples)
    dataset = WaveSource(sidelength= 18)
    # total_batch_size = hparams.batch_size * len(x_true[data_ind])
    bounds = {'x': (-1, 1),
              'y': (-1, 1),
              't': (0, 1)}
    train_dataloader = DataLoader(dataset, batch_size=hparams.batch_size,
                                  shuffle=True,
                                  pin_memory=True, num_workers=0)

    PINN = FCN(hparams.layers, bounds=bounds, device=device, siren=hparams.siren,
               lambda_data=hparams.lambda_data, lambda_pde=hparams.lambda_pde, lambda_bc=hparams.lambda_bc)

    params = list(PINN.dnn.parameters())
    wandb.watch(PINN.dnn, PINN.loss_function, log='all')

    '''Optimization'''
    optimizer = torch.optim.Adam(params, hparams.lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
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

    gf = GreensFunAnalytical(1, 1, 1000, 100, 1e-2)
    analytical_sol = gf.calculate()
    xy_plt, t = gf.return_points()
    t_indx_plt = (1000 * np.linspace(0.1, .5, 5)).astype(int)
    PINN.dnn.train()
    # t_indx_plt = (fs * np.array([0.005, 0.02, 0.035, 0.05, 0.1])).astype(int)
    # t_indx_plt = (fs * np.array([0.005, 0.01, 0.015])).astype(int)
    xyt_plt, p_plt = construct_input_vec(analytical_sol, xy_plt[0], xy_plt[1], t, t_ind=t_indx_plt)
    for epoch in range(max(0, last_epoch), train_epochs):
        for i, batch in enumerate(train_dataloader):
            model_input, gt = batch
            # p_test = batch['pressure_all']
            optimizer.zero_grad()
            loss_total, loss_dir, loss_neu, loss_wave = PINN.SGD_step(model_input, gt)
            optimizer.step()
            if steps % 20 == 0:
                wandb.log({"total_loss": loss_total,
                           "NeumanBC_loss": loss_neu,
                           "DirichletBC_loss": loss_dir,
                           "WaveEq_loss" : loss_wave})
            if steps % int(100) == 0:
                fig = plot_results(xyt_plt, p_plt.reshape(5, -1), PINN)
                wandb.log({"Sound_Fields": wandb.Image(fig)})
                plt.close('all')
            if steps % int(500) == 0:
                checkpoint_path = "{}/PINN_{:08d}".format(checkpoint_dir, steps)
                save_checkpoint(checkpoint_dir,
                                checkpoint_path,
                                {
                                    "net": PINN.dnn.state_dict(),
                                    "optim": optimizer.state_dict(),
                                    "steps": steps,
                                    "epoch": epoch,
                                },
                                remove_below_step=steps // 2
                                )
            steps += 1
            print(f'\repochs: {epoch + 1} total steps: {steps}, loss: {loss_total:.3}', end='',
                  flush=True)


if __name__ == "__main__":
    train_PINN()
# train_PINN()
