import torch
import wandb
from torch.utils.data import DataLoader
from aux_functions import (FCN, PINNDataset, construct_input_vec,
                           plot_results, scan_checkpoint, save_checkpoint,
                           load_checkpoint, get_measurement_vectors,
                           standardize_rirs, normalize_rirs,
                           unit_norm_normalization, maxabs_normalize_rirs)
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
    help="Directory for saving model checkpoints"
)
@click.option(
    "--train_epochs",
    default=1e8,
    type=int,
    help="Number of epochs for which to train PINN (in total)"
)
@click.option(
    "--siren",
    default=True,
    type=bool,
    help="Use sinusoidal activations"
)
@click.option(
    "--real_data",
    default=True,
    type=bool,
    help="Use measurement data"
)
@click.option(
    "--standardize_data",
    default=False,
    type=bool,
    help="Standardize measurement data with global mean and std"
)
@click.option(
    "--lambda_pde",
    default=1e-1,
    type=float,
    help="Lagrangian operator of PDE loss term"
)
@click.option(
    "--lambda_ic",
    default=0.,
    type=float,
    help="Lagrangian operator of initial conditions loss term"
)
@click.option(
    "--lambda_bc",
    default=0.,
    type=float,
    help="Lagrangian operator of boundary conditions loss term")
@click.option(
    "--freq_batch",
    default=120,
    type=int,
    help="Batch size for time domain samples Nt (e.g. total data size = Nx x Ny x Nf)")
@click.option(
    "--n_pde_samples",
    default=15000,
    type=int,
    help="Number of cartesian coordinate samples for satisfying PDE at each iteration")
@click.option(
    "--lr",
    default=2e-5,
    type=float,
    help="(Initial) learning rate value")
@click.option(
    "--scheduler_step",
    default=2e4,
    type=int,
    help="Learning rate scheduler step size (iterations) after which learning rate decreases (set to 0 for no "
         "scheduler)")
@click.option(
    "--max_freq",
    default=3e3,
    type=float,
    help="Max frequency in Hertz, to which the network will learn the Helmholtz equation")
@click.option(
    "--loss_fn", default='mae',
    type=click.Choice(['MAE', 'MSE'], case_sensitive=False),
    help="Loss function to use, (Mean Square Error or Mean Absolute Error)")
@click.option(
    "--max_f_counter", default=-1,
    type=int,
    help="set to -1 to disable seq2seq or curriculum training, otherwise set to i.e. 1e5 to train with"
         "curriculum or seq2seq training until that iteration")
@click.option(
<<<<<<< HEAD
=======
    "--n_mics_per_dimension", default=10,
    type=int,
    help="Number of microphones per dimension (e.g. nxn)")
@click.option(
>>>>>>> main
    "--n_hidden_layers", default=4,
    type=int,
    help="Number of hidden layers")
@click.option(
    "--curriculum_training", default=True,
    type=bool,
    help="Increase time vector (training and PDE data) incrementally in a linear manner")
@click.option(
<<<<<<< HEAD
=======
    "--adaptive_pde_weight", default=False,
    type=bool,
    help="Learnable weight for PDE term, if true, pde_weight_decay_iters is set to null")
@click.option(
    "--pde_weight_decay_iters", default=300000,
    type=int, help="Number of iterations over which PDE loss weight will decay (exponentially) "
                   "to zero from initial value")
@click.option(
>>>>>>> main
    "--map_input", default=True,
    type=bool,
    help="Map all collocation points (e.g. x,y,t) to -1, 1")
def train_PINN(data_dir, checkpoint_dir, train_epochs, siren, real_data,
               standardize_data, n_hidden_layers, lambda_pde, lambda_ic, lambda_bc, freq_batch,
               n_pde_samples, lr, scheduler_step, max_freq, loss_fn, max_f_counter,
<<<<<<< HEAD
               curriculum_training, map_input):
=======
               curriculum_training, map_input, n_mics_per_dimension, pde_weight_decay_iters,
               adaptive_pde_weight):
>>>>>>> main
    config = {
        'max_freq': max_freq,
        'check_point_dir': checkpoint_dir,
        'train_epochs': train_epochs,
        'lr': lr,
        'scheduler_step': scheduler_step,
        'n_hidden_layers': n_hidden_layers,
        'freq_batch': freq_batch,
        'siren': siren,
        'lambda_data': 1.,
        'lambda_pde': lambda_pde,
        'lambda_ic': lambda_ic,
        'lambda_bc': lambda_bc,
        'n_pde_samples': n_pde_samples,
        'real_data': real_data,
        'standardize_data': standardize_data,
        'loss_fn': loss_fn,
        'max_f_counter': max_f_counter,
        'curriculum_training': curriculum_training,
<<<<<<< HEAD
        'map_input' : map_input
    }
    wandb.init(config=config, project="PINN_sound_field_Helmholtz")
=======
        'n_mics_per_dimension': n_mics_per_dimension,
        'map_input': map_input,
        'pde_weight_decay_iters': pde_weight_decay_iters,
        'adaptive_pde_weight': adaptive_pde_weight
    }

    if config['adaptive_pde_weight']:
        config['pde_weight_decay_iters'] = 0

    wandb.init(config=config, project="PINN_sound_field_Helmholtz",
               allow_val_change=True)
>>>>>>> main

    hparams = wandb.config
    if hparams.real_data:
        # filename = data_dir + '/SoundFieldControlPlanarDataset_src2.h5'
        filename = data_dir + '/SoundFieldControlPlanarDataset.h5'
    else:
        filename = data_dir + '/ISM_sphere.npz'
<<<<<<< HEAD
    refdata, fs, grid, measureddata, grid_measured, c = get_measurement_vectors(filename,
                                                                                real_data=hparams.real_data,
                                                                                subsample_points=10)  # per dimension

    # %%
    """Training Data"""
    freq_vec = np.fft.rfftfreq(measureddata.shape[-1], d = 1/fs)
    f_max_ind = np.argmin(freq_vec < max_freq)
    data = np.fft.rfft(measureddata)[:, :f_max_ind]  # truncate
    refdata = np.fft.rfft(refdata)[:, :f_max_ind] # truncate
=======
    refdata, fs, grid, measureddata, \
    grid_measured, c, source_pos, fnyq = get_measurement_vectors(filename,
                                                           real_data=hparams.real_data,
                                                           subsample_points=hparams.n_mics_per_dimension)  # per dim

    # %%
    """Training Data"""
    fnyq_avg = np.mean(fnyq)
    freq_vec = np.fft.rfftfreq(measureddata.shape[-1], d=1 / fs)
    f_max_ind = np.argmin(freq_vec < fnyq_avg)
    data = np.fft.rfft(measureddata)[:, :f_max_ind]
    refdata = np.fft.rfft(refdata)[:, :f_max_ind]
    # f_max_ind = np.argmin(freq_vec < max_freq)
    # data = np.fft.rfft(measureddata)[:, :f_max_ind]  # truncate
    # refdata = np.fft.rfft(refdata)[:, :f_max_ind]  # truncate
>>>>>>> main

    if hparams.standardize_data:
        scaler = standardize_rirs(refdata, device=device)
    else:
        # scaler = normalize_rirs(refdata, device=device)
        # scaler = unit_norm_normalization(refdata, device=device)
        if hparams.map_input:
            l_inf_norm = 0.1
        else:
            l_inf_norm = 1
<<<<<<< HEAD
        scaler = maxabs_normalize_rirs(refdata, device=device, l_inf_norm = l_inf_norm)
=======
        scaler = maxabs_normalize_rirs(refdata, device=device, l_inf_norm=l_inf_norm)
>>>>>>> main

    f_ind = np.arange(0, refdata.shape[-1])
    x_m = grid_measured[0]
    y_m = grid_measured[1]
    x_ref = grid[0]
    y_ref = grid[1]
<<<<<<< HEAD
=======

>>>>>>> main
    # %%

    bounds = {
        'x': (1.1 * x_ref.min(), 1.1 * x_ref.max()),
        'y': (1.1 * y_ref.min(), 1.1 * y_ref.max()),
        'f': (0, hparams.max_freq)}
<<<<<<< HEAD

    PINN = FCN(n_hidden_layers=hparams.n_hidden_layers, bounds=bounds, device=device, siren=hparams.siren,
               lambda_data=hparams.lambda_data, lambda_pde=hparams.lambda_pde, lambda_bc=hparams.lambda_bc,
               c=c, lambda_ic=hparams.lambda_ic, loss_fn=hparams.loss_fn, output_scaler=scaler,
               fs=fs, map_input = hparams.map_input)

    net_params = list(PINN.dnn.parameters())
=======

    PINN = FCN(n_hidden_layers=hparams.n_hidden_layers, bounds=bounds, device=device, siren=hparams.siren,
               lambda_data=hparams.lambda_data, lambda_pde=hparams.lambda_pde, lambda_bc=hparams.lambda_bc,
               c=c, lambda_ic=hparams.lambda_ic, loss_fn=hparams.loss_fn, output_scaler=scaler,
               fs=fs, map_input=hparams.map_input, source_pos = source_pos)

    if hparams.adaptive_pde_weight:
        PINN.lambda_pde = torch.nn.Parameter(torch.FloatTensor(hparams.n_pde_samples,
                                                               1).uniform_(0., hparams.lambda_pde).to(device))
        extra_params = [PINN.lambda_pde]
        lambda_optimizer = torch.optim.Adam(extra_params, 1e-5, betas=(0.9, 0.999),
                                            weight_decay= 1e-9)

>>>>>>> main
    wandb.watch(PINN.dnn, PINN.loss_function, log='all')

    '''Optimization'''
    gamma = 0.9  # final learning rate will be gamma * initial_lr
<<<<<<< HEAD
    optimizer = torch.optim.Adam(net_params, hparams.lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
=======
    net_params = list(PINN.dnn.parameters())

    optimizer = torch.optim.Adam(net_params, hparams.lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=False, weight_decay= 1e-7)
>>>>>>> main
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=hparams.scheduler_step if hparams.scheduler_step > 0 else 1,
                                                gamma=gamma)

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
<<<<<<< HEAD
=======
        if hparams.adaptive_pde_weight:
            lambda_optimizer.load_state_dict(state_dict_pinn["lambda_optim"])
>>>>>>> main

    # Dataset
    dataset = PINNDataset(refdata=refdata, measured_data=data, x_ref=x_ref, y_ref=y_ref,
                          x_m=x_m, y_m=y_m, f=freq_vec, f_ind=f_ind,
                          n_pde_samples=hparams.n_pde_samples, counter=steps + 1,
                          maxcounter=hparams.max_f_counter,
                          curriculum_training=hparams.curriculum_training,
                          batch_size=hparams.freq_batch)
    train_dataloader = DataLoader(dataset, batch_size=hparams.freq_batch,
                                  shuffle=True,
                                  pin_memory=True, num_workers=0)

    PINN.dnn.train()
<<<<<<< HEAD
    f_plt = [250., 500., 1000., 2000., 2999.]
    f_ind_plt = np.array([np.argmin(freq_vec < f_) for f_ in f_plt])
    f_plt = freq_vec[f_ind_plt]
    # adaptive weight for pde
    if hparams.max_f_counter != -1:
        lambda_pde_decay = (1 - .98) ** np.linspace(0, 1, hparams.max_f_counter)
    else:
        lambda_pde_decay = np.ones((1,))
    xyf_plt, p_plt = construct_input_vec(refdata, x_ref, y_ref, freq_vec, f_ind=f_ind_plt)
=======
    f_plt = [125., 250., 500., 1000., 1800.]
    f_ind_plt = np.array([np.argmin(freq_vec < f_) for f_ in f_plt])

    # adaptive weight for pde
    lambda_pde_decay = (1 - .98) ** np.linspace(0, 1, hparams.pde_weight_decay_iters)
    xyf_plt, p_plt = construct_input_vec(refdata, x_ref, y_ref, freq_vec, f_ind=f_ind_plt)

>>>>>>> main
    for epoch in range(max(0, last_epoch), train_epochs):
        for i, batch in enumerate(train_dataloader):
            data_input = batch['collocation_train']
            pde_input = batch['collocation_pde']
            ic_input = batch['collocation_ic']
            p_data = batch['pressure_batch']
            f_lims = batch['f_lims']
            data_loss_weights = batch['data_loss_weights']
            # p_test = batch['pressure_all']
            optimizer.zero_grad()
<<<<<<< HEAD
=======
            if hparams.adaptive_pde_weight:
                lambda_optimizer.zero_grad()
>>>>>>> main
            loss_total, loss_data, loss_pde, loss_bc, loss_ic, norm_ratio, std_ratio, maxabs_ratio = PINN.SGD_step(
                data_input.to(device),
                pde_input.to(device),
                ic_input.to(device),
                p_data.to(device),
                data_loss_weights.to(device)
            )
            optimizer.step()
<<<<<<< HEAD
=======
            if hparams.adaptive_pde_weight:
                lambda_optimizer.step()
>>>>>>> main
            if hparams.scheduler_step > 0:
                scheduler.step()
            if steps % 100 == 0:
                wandb.log({
                    "total_loss": loss_total,
                    "data_loss": loss_data,
                    "PDE_loss": loss_pde,
                    "BCs_loss": loss_bc,
                    "ICs_loss": loss_ic,
                    "norm_ratio": norm_ratio,
                    "std_ratio": std_ratio,
                    "maxabs_ratio": maxabs_ratio})
            if steps % int(1000) == 0:
                fig, errors, avg_snapshot_error = plot_results(xyf_plt, p_plt, PINN)
                wandb.log({"Sound_Fields": wandb.Image(fig)})
                plt.close('all')
                for ii, error in enumerate(errors):
                    wandb.log({
                        f"MSE - t: {xyf_plt[ii, 2, 0]:.3f} s": error,
                        "steps": steps})
                wandb.log({
                    "Average snapshot square error": avg_snapshot_error,
                    "steps": steps})
                wandb.log({
                    "Training (time) window lower bound": f_lims[0].item(),
                    "Training (time) window upper bound": f_lims[1].item(),
                    "steps": steps})
            if steps % int(1000) == 0:
                checkpoint_path = "{}/PINN_{:08d}".format(checkpoint_dir, steps)
                state_dict_ = {
                                    "net": PINN.dnn.state_dict(),
                                    "optim": optimizer.state_dict(),
                                    "steps": steps,
                                    "epoch": epoch,
                                }
                if hparams.adaptive_pde_weight:
                    state_dict_["lambda_optim"] = lambda_optimizer.state_dict()
                save_checkpoint(checkpoint_dir,
                                checkpoint_path,
                                state_dict_,
                                remove_below_step=steps // 2
                                )
<<<<<<< HEAD
            if steps < len(lambda_pde_decay):
                PINN.lambda_pde = hparams.lambda_pde*lambda_pde_decay[steps]
            else:
                PINN.lambda_pde = hparams.lambda_pde*lambda_pde_decay[-1]
            steps += 1
            print(
                f'\repochs: {epoch + 1} total steps: {steps}, loss: {loss_total:.3}, f limits: ({f_lims[0].item():.3}, '
                f'{f_lims[1].item():.3}) Hz, lambda_pde: {PINN.lambda_pde}',
=======
            if np.logical_and(steps < hparams.pde_weight_decay_iters, hparams.pde_weight_decay_iters == -1):
                PINN.lambda_pde = hparams.lambda_pde * lambda_pde_decay[steps]

            steps += 1
            if hparams.adaptive_pde_weight:
                lambda_pde_print = PINN.lambda_pde.mean().data
            else:
                lambda_pde_print = PINN.lambda_pde
            print(
                f'\repochs: {epoch + 1} total steps: {steps}, loss: {loss_total:.3}, f limits: ({f_lims[0].item():.3}, '
                f'{f_lims[1].item():.3}) Hz, lambda_pde: {lambda_pde_print}',
>>>>>>> main
                end='',
                flush=True)


if __name__ == "__main__":
    train_PINN()
