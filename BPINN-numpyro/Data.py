from pyDOE import lhs
import numpy as np
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from utils_soundfields import plot_sf

def create_dataset(params, time_truncation = 0.1):
    """Creates a dataset based on solving the wave equation on given parameters.

    Args:
        params: Dictionary containing relevant parameters.

    Returns:, collocation_data_m, collocation_pde, lookup_reference, lookup_measured
        collocation_data_ref : Reference data inputs  [x_coord, y_coord, t_time]
        collocation_pde : PDE inputs  [x_coord, y_coord, t_time]
        collocation_data_m : Data inputs  [x_coord, y_coord, t_time]
        lookup_reference: Target for reference pressure field and indices of each axis (grid x time)
        lookup_measured: Target for measured pressure field and indices of each axis (grid x time)
    """
    def spatiotemporal_matching(lookup, x, column = 2):
        x_out = []
        for indx in lookup[:, column]:
            x_out.append(x[int(indx)])
        return np.array(x_out)
    # Unpack parameters
    n_collocation = params['n_collocation']
    x_min, x_max = params['x_bounds']
    y_min, y_max = params['y_bounds']
    t_min, t_max = params['t_bounds']

    # Define upper and lower bounds
    lb = np.array([x_min, y_min, t_min])[..., None]
    ub = np.array([x_max, y_max, t_max])[..., None]

    scaling = lambda x: 2.0 * (x - lb) / (ub - lb) - 1.0
    # load measurements
    p = Path(params['data_path'])
    data = np.load(p.resolve())
    # keys = [key for key in data.keys()]
    p_reference = data['reference_data']
    p_measured = data['array_data']
    fs = data['fs']
    t_data = np.linspace(0., len(p_reference)/fs, len(p_reference))
    grid_measured = data['grids_measured']
    grid_reference = data['grid_reference']

    p_reference = p_reference[:, :int(fs* time_truncation)]
    p_measured = p_measured[:, :int(fs* time_truncation)]
    t_data = t_data[:int(fs* time_truncation)]

    pxy_ref, pt_ref = np.meshgrid(np.arange(p_reference.shape[1]), np.arange(p_reference.shape[0]))
    lookup_reference = np.vstack((p_reference.ravel(),pt_ref.ravel().view(int), pxy_ref.ravel().view(int))).T
    x_ref = spatiotemporal_matching(lookup_reference, grid_reference[0], column = 1)
    y_ref = spatiotemporal_matching(lookup_reference, grid_reference[1], column = 1)
    t_ref = spatiotemporal_matching(lookup_reference, t_data, column= 2)
    collocation_data_ref =  np.vstack((x_ref,y_ref,t_ref))

    pxy_m, pt_m = np.meshgrid(np.arange(p_measured.shape[1]), np.arange(p_measured.shape[0]))
    lookup_measured =  np.vstack((p_measured.ravel(), pt_m.ravel(),pxy_m.ravel())).T
    x_m = spatiotemporal_matching(lookup_measured, grid_measured[0], column = 1)
    y_m = spatiotemporal_matching(lookup_measured, grid_measured[1], column = 1)
    t_m = spatiotemporal_matching(lookup_measured, t_data, column= 2)
    collocation_data_m =  np.vstack((x_m,y_m,t_m))


    grid_pde = 2 * (lhs(2, n_collocation)) / 1 - 1
    t_pde = lhs(1, n_collocation)

    collocation_pde = np.hstack((grid_pde[:, 0, None], grid_pde[:, 1, None], t_pde)).T

    return collocation_data_ref, collocation_data_m, collocation_pde, lookup_reference, lookup_measured

def plot_results(collocation_all, lookup_all, PINN):
    collocation_data, pressure = get_plot_groundtruths(collocation_all,lookup_all)

    Nplots = collocation_data.shape[0]
    Pred_pressure = []
    error_vecs = []
    relative_errors = []
    tfnp = lambda x: torch.from_numpy(x).float().to(PINN.device)


    for n in range(Nplots):
        error_vec, relative_error, p_pred = PINN.test(tfnp(collocation_data[n]), tfnp(pressure[n]))
        Pred_pressure.append(p_pred)
        error_vecs.append(error_vec)
        relative_errors.append(relative_error)
    fig, axes = plt.subplots(nrows=3, ncols=Nplots, sharex=True, sharey=True)
    error_vec_minmax = (np.array(error_vecs).min(), (1.2*np.array(error_vecs).min()))
    p_pred_minmax = (np.array(Pred_pressure).min(), np.array(Pred_pressure).min()+ 1.)
    p_true_minmax = (pressure.min(), pressure.max())
    for i, ax in enumerate(axes[0]):
        if i == 2:
            name = 'Predicted - \n'
        else:
            name = ''
        ax, im = plot_sf(Pred_pressure[i], collocation_data[i, :, 0], collocation_data[i, :, 1],
                        ax=ax, name= name +'t = {:.2f}s'.format(collocation_data[i, 2, 0]),
                        clim= p_pred_minmax)
        if i != 0:
            ax.set_ylabel('')
    for i, ax in enumerate(axes[1]):
        if i == 2:
            name = 'True'
        else:
            name = ''
        ax, im2 = plot_sf(pressure[i], collocation_data[i, :, 0], collocation_data[i,  :, 1],
                        ax=ax, name = name,
                        clim = p_true_minmax)
        if i != 0:
            ax.set_ylabel('')
    for i, ax in enumerate(axes[2]):
        if i == 2:
            name = 'Relative Error'
        else:
            name = ''
        ax, im3 = plot_sf(error_vecs[i], collocation_data[i, :,  0], collocation_data[i, :,  1],
                        ax=ax, name = name, clim = error_vec_minmax, cmap = 'hot')
        if i != 0:
            ax.set_ylabel('')
    fig.subplots_adjust(right=0.8)
    # pressure colorbar
    cbar_ax = fig.add_axes([0.82, 0.71, 0.02, 0.2])
    fig.colorbar(im, cax=cbar_ax)
    cbar_ax2 = fig.add_axes([0.82, 0.41, 0.02, 0.2])
    fig.colorbar(im2, cax=cbar_ax2)
    # error colorbar
    cbar_ax3 = fig.add_axes([0.82, 0.11, 0.02, 0.2])
    fig.colorbar(im3, cax=cbar_ax3)

    t_s = list(collocation_data[j, 0, 2] for j in range(Nplots))
    return fig, np.array(relative_errors), t_s

def get_plot_groundtruths(collocation_data, pressure_data):

    collocation_new = []
    pressure = []
    t = [100, 150, 300, 500, 750]
    for tt in t:
        collocation_new.append(collocation_data[tt::800, :].cpu().detach().numpy())
        pressure.append(pressure_data[tt::800].cpu().detach().numpy())

    return np.array(collocation_new), np.array(pressure)