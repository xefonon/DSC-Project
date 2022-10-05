from pyro.infer.autoguide.guides import (AutoDiagonalNormal, AutoMultivariateNormal,
                                         AutoNormal)
from pyro.infer import HMC, MCMC, NUTS, SVI, Predictive, Trace_ELBO, TraceMeanField_ELBO
from pyro.optim import Adam, ClippedAdam
from pyro.infer.autoguide import init_to_feasible
from pyro.infer import Predictive
from aux_functions import scan_checkpoint, load_checkpoint, save_checkpoint
import os
import pyro
from Data import create_dataset, plot_results
from BPINN import Model_BPINN
import torch
import numpy as np
import wandb
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

checkpoint_dir = './BPINN_checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

if os.path.isdir(checkpoint_dir):
    cp_pinn = scan_checkpoint(checkpoint_dir, "PINN_")

nn_params = {'h0': 3,
             'h1': 256,
             'h2': 256,
             'h3': 256,
             'h4': 256,
             'sigma': .1,
             'device' : device}


data_params = {'data_path': '../Data/ISM_sphere.npz',
               'n_collocation': 20000,
               'x_bounds': (-1, 1),
               'y_bounds': (-1, 1),
               't_bounds': (0, .1*343),
               'speed_of_sound': 343.,
               'batch_size' : 10000}

run_params = {'n_steps' : 100000,
              'initial_lr' : 0.0002,
              'gamma' : 0.1}

config = {**nn_params, **data_params, **run_params}
wandb.init(config=config, project="BPINN_sound_field")
# model
model = Model_BPINN(data_params, nn_params).to(nn_params['device'])
wandb.watch(model.net, log='all')

# Define the number of optimization steps
n_steps = run_params['n_steps']
# Setup the optimizer
initial_lr = run_params['initial_lr']
gamma = run_params['gamma'] # final learning rate will be gamma * initial_lr
lrd = gamma ** (1 / n_steps)
optimizer = pyro.optim.ClippedAdam({'lr': initial_lr, 'lrd': lrd})
# adam_params = {"lr": 0.00002}
# optimizer = Adam(adam_params)

steps = 0
if cp_pinn is None:
    last_epoch = -1
else:
    state_dict_pinn = load_checkpoint(cp_pinn, nn_params['device'])
    model.net.load_state_dict(state_dict_pinn["net"])
    steps = state_dict_pinn["steps"] + 1

# Prepare data
collocation_data_ref, collocation_data_m, collocation_pde, lookup_reference, lookup_measured = create_dataset(data_params)

Xdata = torch.from_numpy(collocation_data_m.T).float().to(device)
Xref = torch.from_numpy(collocation_data_ref.T).float().to(device)
Xpde = torch.from_numpy(collocation_pde.T).float().to(device)
P_m = torch.from_numpy(lookup_measured[:, 0]).float().to(device)
P_ref = torch.from_numpy(lookup_reference[:, 0]).float().to(device)
F = torch.zeros(Xpde.shape[0], 1).to(device)
# Define guide function
guide = AutoNormal(model, init_loc_fn=init_to_feasible)
# Reset parameter values
pyro.clear_param_store()

# Setup the inference algorithm
elbo = Trace_ELBO(num_particles=5)
svi = SVI(model, guide, optimizer, loss=elbo)

# Do gradient steps
for step in range(n_steps):
    elbo = svi.step(Xdata, Xpde,  P_m, F)
    if step % 2000 == 0:
        valid_elbo = svi.evaluate_loss(Xdata = Xref, p_measured = P_ref)
        print("[%d] Validation ELBO: %.1f" % (step, valid_elbo))
        wandb.log({"Validation_ELBO": valid_elbo})
    if step % 2000 == 0:
        fig, errors, t_s = plot_results(Xref, P_ref, model)
        wandb.log({"Sound_Fields": wandb.Image(fig)})
        plt.close('all')
        for ii, error in enumerate(errors):
            wandb.log({
                f"error - t: {t_s[ii]:.3f} s": error,
                "steps": step})
    if step % int(10000) == 0:
        checkpoint_path = "{}/PINN_{:08d}".format(checkpoint_dir, steps)
        # save_checkpoint(checkpoint_dir,
        #                 checkpoint_path,
        #                 {
        #                     "net": model.net.state_dict(),
        #                     "steps": steps,
        #                 },
        #                 remove_below_step=steps // 2
        #                 )
        torch.save({"model": model.state_dict(), "guide": guide}, checkpoint_path)
        pyro.get_param_store().save(checkpoint_path + "_pyromodelparams")

    if step % 500 == 0:
        print("[%d] ELBO: %.1f" % (step, elbo))
        wandb.log({"ELBO": elbo})

    steps += 1
# %%

num_samples = 2
predictive = Predictive(model, guide=guide, num_samples=num_samples, return_sites=["x", "xx"])
samples_svi = predictive(Xref[:10000])
#%%
pred_mean = samples_svi["obs1"].detach().mean(axis=0).squeeze(-1)
pred_std = samples_svi["obs1"].detach().std(axis=0).squeeze(-1)

# saved_model_dict = torch.load("mymodel.pt")
# model.load_state_dict(saved_model_dict['model'])
# guide = saved_model_dict['guide']
# pyro.get_param_store().load("mymodelparams.pt")
