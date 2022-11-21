import sys

sys.path.append('../')
import torch
import torch.autograd as autograd  # computation graph
from numpy.lib.stride_tricks import sliding_window_view

# from torch import Tensor  # tensor node in the computation graph
import torch.nn as nn  # neural networks
import numpy as np
from torch.utils.data import Dataset
from utils_soundfields import plot_sf
# from tueplots import axes, bundles
import matplotlib.pyplot as plt
import os
import glob
import re
from pyDOE import lhs
# from SIREN import Siren, SirenNet
from modules_meta import SingleBVPNet
import h5py
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import librosa

def subsample_gridpoints(grid, subsample=5):
    r0 = grid.mean(axis=-1)
    tempgrid = grid - r0[:, None]
    xmin, xmax = round(tempgrid[0].min(), 3), round(tempgrid[0].max(), 3)
    ymin, ymax = round(tempgrid[1].min(), 3), round(tempgrid[1].max(), 3)
    newgrid = reference_grid(subsample, xmin, xmax)
    newgrid += r0[:2, None]
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(grid[:2].T)
    distances, indices = nbrs.kneighbors(newgrid.T)
    return grid[:, indices.squeeze(-1)], indices.squeeze(-1)


def reference_grid(steps, xmin=-.7, xmax=.7):
    x = np.linspace(xmin, xmax, steps)
    y = np.linspace(xmin, xmax, steps)
    # z = tf.zeros(shape = (steps,))
    X, Y = np.meshgrid(x, y)
    gridnew = np.vstack((X.reshape(1, -1), Y.reshape(1, -1)))
    return gridnew


def speed_of_sound(T):
    """
    speed_of_sound(T)
    Caculate the adiabatic speed of sound according to the temperature.
    Parameters
    ----------
    T : double value of temperature in [C].
    Returns
    -------
    c : double value of speed of sound in [m/s].
    """
    c = 20.05 * np.sqrt(273.15 + T)
    return c


def load_measuremenf_data(filename):
    with h5py.File(filename, "r") as f:
        data_keys = f.keys()
        meta_data_keys = f.attrs.keys()
        data_dict = {}
        for key in data_keys:
            data_dict[key] = f[key][:]
        for key in meta_data_keys:
            data_dict[key] = f.attrs[key]
        f.close()
    return data_dict


class standardize_rirs:
    def __init__(self, data, device='cuda'):
        self.data = data
        self.tfnp = lambda x: torch.from_numpy(x).float().to(device)
        self.tfnp_cmplx = lambda x: torch.from_numpy(x).type(torch.complex64)
        self.mean = self.tfnp_cmplx(data.mean()[None, None])
        self.std = self.tfnp(data.std()[None, None])

    def forward_rir(self, input):
        return (input - self.mean) / self.std

    def backward_rir(self, input):
        return input * self.std + self.mean


# class standardize_rirs:
#     def __init__(self, data, device='cuda'):
#         self.data = data
#         self.tfnp = lambda x: torch.from_numpy(x).float().to(device)
#
#         self.mean = self.tfnp(data.mean(axis=0)[None, :])
#         self.std = self.tfnp(data.std(axis=0)[None, :])
#
#     def forward_rir(self, input):
#         return (input - self.mean) / self.std
#
#     def backward_rir(self, input):
#         return input * self.std + self.mean

class unit_norm_normalization:
    def __init__(self, data, device='cuda'):
        self.data = data
        self.tfnp = lambda x: torch.from_numpy(x).float().to(device)
        self.l2_norm = lambda x: np.linalg.norm(x)

        self.norm = self.tfnp(self.l2_norm(data)[None, None])

    def forward_rir(self, input):
        return input / self.norm

    def backward_rir(self, input):
        return input * self.norm


class normalize_rirs:
    def __init__(self, data, device='cuda'):
        self.data = data
        self.tfnp = lambda x: torch.from_numpy(x).float().to(device)
        self.lb = data.min()
        self.ub = data.max()
        self.scaling = lambda x: 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        self.unscaling = lambda x: (x + 1) * (self.ub - self.lb) / 2. + self.lb

        # self.norm = self.tfnp(self.maxabs(data)[None, None]/.95)

    def forward_rir(self, input):
        return self.scaling(input)

    def backward_rir(self, input):
        return self.unscaling(input)


class maxabs_normalize_rirs:
    def __init__(self, data, device='cuda', l_inf_norm = 0.1):
        self.data = data
        self.tfnp = lambda x: torch.from_numpy(x).float().to(device)
        self.tfnp_cmplx = lambda x: torch.from_numpy(x).type(torch.complex64)

        self.maxabs = lambda x: np.max(abs(x))
        self.l_inf_norm = l_inf_norm

        self.norm = self.tfnp(self.maxabs(data)[None, None])

    def forward_rir(self, input):
        return self.l_inf_norm*input / (self.norm)

    def backward_rir(self, input):
        return (1/self.l_inf_norm)*input * self.norm


def get_measurement_vectors(filename, real_data=True, subsample_points=10):
    if real_data:
        data_dict = load_measuremenf_data(filename)
        refdata = data_dict['RIRs_bottom']
        temperature = data_dict['temperature']
        c = speed_of_sound(temperature)
        fs = data_dict['fs']
        refdata = librosa.resample(refdata, fs, 8000)
        fs = 8000
        grid = data_dict['grid_bottom']
        measureddata = refdata
        # grid_measured = data_dict['grid_bottom']
        grid -= grid.mean(axis=-1)[:, None]
        grid_measured, indcs = subsample_gridpoints(grid, subsample=subsample_points)
        measureddata = measureddata[indcs]

    else:
        p = Path(filename)
        if not p.exists():
            p = Path('./Data/ISM_sphere.npz')
        data = np.load(p.resolve())
        keys = [key for key in data.keys()]
        print("datafile keys: ", keys)
        refdata = data['reference_data']
        fs = data['fs']
        grid = data['grid_reference']
        measureddata = data['array_data']
        grid_measured = data['grid_measured']
        c = 343.
    return refdata, fs, grid, measureddata, grid_measured, c


def save_checkpoint(directory, filepath, obj, remove_below_step=None):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")
    if remove_below_step is not None:
        print("\nRemoving checkpoints below step ", remove_below_step)
        remove_checkpoint(directory, remove_below_step)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + "????????")
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def remove_checkpoint(cp_dir, delete_below_steps=1000):
    filelist = [f for f in os.listdir(cp_dir) if f.startswith("PINN")]
    for f in filelist:
        prefix, number, extension = re.split(r'(\d+)', f)
        if int(number) < delete_below_steps:
            os.remove(os.path.join(cp_dir, f))


def construct_input_vec(TFdata, x_true, y_true, f, data_ind=None, f_ind=None):
    if data_ind is not None:
        TFdata = TFdata[data_ind]
        x_true = x_true[data_ind]
        y_true = y_true[data_ind]
    if f_ind is not None:
        TFdata = TFdata[:, f_ind]
        f = f[f_ind]
    collocation = []
    for i in range(len(f)):
        ff = np.repeat(f[i], len(x_true))
        collocation.append(np.stack([x_true, y_true, ff], axis=0))
    return np.array(collocation), TFdata


def plot_results(collocation_data, rirdata, PINN):
    Nplots = collocation_data.shape[0]
    Pred_pressure = []
    error_vecs = []
    mean_square_errors = []
    square_errors = []
    tfnp = lambda x: torch.from_numpy(x).float().to(PINN.device)
    tfnp_cmplx = lambda x: torch.from_numpy(x).type(torch.complex64)

    for n in range(Nplots):
        error_vec, mean_square_error, square_error, p_pred = PINN.test(tfnp(collocation_data[n]), tfnp_cmplx(rirdata[:, n]))
        Pred_pressure.append(p_pred)
        error_vecs.append(error_vec)
        mean_square_errors.append(mean_square_error)
        square_errors.append(square_error)
    fig, axes = plt.subplots(nrows=3, ncols=Nplots, sharex=True, sharey=True)
    error_vec_minmax = (np.array(error_vecs).min(),
                        np.minimum(np.array(error_vecs).max(),
                                   np.maximum(1., np.array(error_vecs).min() + 1e-5)))
    p_pred_minmax = (np.array(Pred_pressure).real.min(), np.array(Pred_pressure).real.max() + np.finfo(np.float32).eps)
    p_true_minmax = (rirdata.real.min(), rirdata.real.max() + np.finfo(np.float32).eps)
    for i, ax in enumerate(axes[0]):
        if i == 2:
            name = 'Predicted - \n'
        else:
            name = ''
        ax, im = plot_sf(Pred_pressure[i], collocation_data[i, 0], collocation_data[i, 1],
                         ax=ax, name=name + 'f = {:.1f}Hz'.format(collocation_data[i, 2, 0]),
                         clim=p_pred_minmax, normalise=False)
        if i != 0:
            ax.set_ylabel('')
    for i, ax in enumerate(axes[1]):
        if i == 2:
            name = 'True'
        else:
            name = ''
        ax, im2 = plot_sf(rirdata[:, i], collocation_data[i, 0], collocation_data[i, 1],
                          ax=ax, name=name,
                          clim=p_true_minmax, normalise=False)
        if i != 0:
            ax.set_ylabel('')
    for i, ax in enumerate(axes[2]):
        if i == 2:
            name = 'Relative Error'
        else:
            name = ''
        ax, im3 = plot_sf(error_vecs[i], collocation_data[i, 0], collocation_data[i, 1],
                          ax=ax, name=name, clim=error_vec_minmax, cmap='hot', normalise=False)
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

    return fig, np.array(mean_square_errors), np.array(square_errors).sum(axis=0).mean(),


#  Deep Neural Network
class DNN(nn.Module):
    def __init__(self, n_hidden_layers, lb, ub, siren=True, scale_input = True, out_features = 2):
        super().__init__()  # call __init__ from parent class
        self.siren = siren
        'activation function'
        if self.siren:
            self.activation = nn.Identity()
        else:
            self.activation = nn.Tanh()
        self.lb = lb
        self.ub = ub
        self.n_hidden_layers = n_hidden_layers
        self.scaling = lambda x: 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        self.scale_input = scale_input
        self.out_features = out_features
        'Initialize neural network as a list using nn.Modulelist'
        if self.siren:
            # self.net = SirenNet(
            #                 dim_in = layers[0],               # input dimension, ex. 2d coor
            #                 dim_hidden = 256,                 # hidden dimension
            #                 dim_out = out_features,                      # output dimension, ex. rgb value
            #                 num_layers = 5,                   # number of layers
            #                 final_activation = nn.Identity(), # activation of final layer (nn.Identity() for direct output)
            #                 w0_initial = 30.,                 # different signals may require different omega_0 in the first layer - this is a hyperparameter
            #                 w0 = 1.
            #             )
            self.net = SingleBVPNet(out_features=self.out_features, in_features=3, hidden_features= 512,
                                    num_hidden_layers=n_hidden_layers)
        else:
            layers = np.array([3] + n_hidden_layers*[100] + [1])
            self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
            'Xavier Normal Initialization'
            for i in range(len(layers) - 1):
                nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
                # set biases to zero
                nn.init.zeros_(self.linears[i].bias.data)

    'foward pass'
    def forward(self, input):
        # batch_size = input.shape[1]
        g = input.clone()
        x, y, f = g[:, 0, :].flatten(), g[:, 1, :].flatten(), g[:, 2, :].flatten()

        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)
        if torch.is_tensor(y) != True:
            y = torch.from_numpy(y)
        if torch.is_tensor(f) != True:
            f = torch.from_numpy(f)

        # convert to float
        x = x.float()
        y = y.float()
        f = f.float()

        # preprocessing input - feature scaling function
        input_preprocessed = torch.stack((x, y, f), dim=-1).view(-1, 3)
        if self.scale_input:
            z = self.scaling(input_preprocessed)
        else:
            z = input_preprocessed
        if self.siren:
            # p_out = torch.view_as_complex(self.net(z)).unsqueeze(1)  # z: [batchsize x 3]
            p_out = self.net(z)  # z: [batchsize x 3]
        else:
            for i in range(len(self.layers) - 2):
                z = self.linears[i](z)
                z = self.activation(z)

            p_out = self.linears[-1](z)
            # p_out = torch.view_as_complex(p_out).unsqueeze(1)
        return p_out

def distance_between(s, r):
    """Distance of all combinations of locations in s and r
    Args:
        s (ndarray [N, 2]): cartesian coordinates of s
        r (ndarray [M, 2]): cartesian coordinates of r
    Returns:
        ndarray [M, N]: distances in meters
    """
    s = torch.atleast_2d(s)
    r = torch.atleast_2d(r)
    if s.shape[-1]!=2:
        s = s.T
    if r.shape[-1]!=2:
        r = r.T
    return torch.linalg.norm(s[None, :] - r[:, None], axis=-1)

def plane_wave(t, c, fs, source_coords, receiver_coords):
    """

    Parameters
    ----------
    t - instanteneous time (scalar)
    c
    fs
    source_coords
    receiver_coords

    Returns
    -------

    """
    with torch.no_grad():
        receiver_coords = receiver_coords / torch.linalg.norm(receiver_coords, axis = 0)
    d = distance_between(source_coords, receiver_coords)
    Dt = t[:, None] - (d / c)
    return torch.sinc(fs * Dt)

#  PINN
# https://github.com/alexpapados/Physics-Informed-Deep-Learning-Solid-and-Fluid-Mechanics
class FCN():
    def __init__(self,
                 bounds,
                 n_hidden_layers = 4,
                 device=None,
                 siren=True,
                 lambda_data=1.,
                 lambda_pde=1e-4,
                 lambda_bc=1e-2,
                 lambda_ic=1e-2,
                 loss_fn='mae',
                 c=343.,
                 output_scaler=None,
                 fs = 48e3,
                 map_input = True):
        if device is None:
            self.device = 'cpu'
        else:
            self.device = device
        if loss_fn in ['mse', 'MSE']:
            self.loss_function = nn.MSELoss(reduction='mean')
            # self.loss_function = lambda y_hat, y : ((torch.abs(y - y_hat)**2)/torch.abs(y)**2).mean()
            self.loss_function_pde =  nn.MSELoss(reduction='mean')
        else:
            self.loss_function = nn.L1Loss(reduction='mean')
            self.loss_function_pde =  nn.L1Loss(reduction='mean')
            # self.loss_function = nn.L1Loss(reduction='sum')
        'Initialize iterator'
        self.iter = 0
        self.fs = fs
        self.lambda_data = lambda_data
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.lambda_ic = lambda_ic
        self.siren = siren
        'speed of sound'
        self.c = c
        self.output_scaler = output_scaler if output_scaler is not None else nn.Identity()

        (self.xmin, self.xmax) = bounds['x']
        (self.ymin, self.ymax) = bounds['y']
        (self.fmin, self.fmax) = bounds['f']

        self.fmax /= self.c
        self.fmin /= self.c
        self.lb = torch.Tensor([self.xmin, self.ymin, self.fmin]).to(self.device)
        self.ub = torch.Tensor([self.xmax, self.ymax, self.fmax]).to(self.device)
        'Call our DNN'
        self.dnn = DNN(n_hidden_layers, lb=self.lb, ub=self.ub, siren=siren,
                       scale_input = map_input).to(device)
        'test with cosine similarity loss'
        self.cosine_sim = nn.CosineEmbeddingLoss()

    def cylindrical_coords(self, input):
        x, y, t = input[:, 0, :].flatten(), input[:, 1, :].flatten(), input[:, 2, :].flatten()
        r = torch.sqrt(x ** 2 + y ** 2)
        phi = torch.atan2(y, x)
        return r, phi

    def loss_data(self, input, pm, data_loss_weights = None):
        g = input.clone()
        g = self.scale_f(g)
        g.requires_grad = True
        pressure = self.dnn(g)
        data_loss_weights = torch.stack((data_loss_weights, data_loss_weights), axis = -1)
        loss_u = self.loss_function(data_loss_weights*pressure.unsqueeze(0), data_loss_weights*torch.view_as_real(pm))
        # loss_u = self.loss_function(pressure, pm)
        norm_ratio = pm.norm()/pressure.norm()
        std_ratio = pm.std()/pressure.std()
        maxabs_ratio = abs(pm).max()/abs(pressure).max()

        return loss_u, norm_ratio, std_ratio, maxabs_ratio

    def loss_PDE(self, input):

        g = input.clone()
        g = self.scale_f(g)
        g.requires_grad = True

        k = 2*np.pi*g[:, 2]#/self.c
        pnet = self.dnn(g)

        # pnet = self.output_scaler.backward_rir(pressure)

        p_r_re = autograd.grad(pnet[0], g, torch.ones_like(pnet[0]),
                                create_graph=True)[0]
        p_r_im = autograd.grad(pnet[1], g, torch.ones_like(pnet[1]),
                                create_graph=True)[0]
        p_rr_re = \
            autograd.grad(p_r_re.view(-1, 1), g, torch.ones(input.view(-1, 1).shape).to(self.device),
                          create_graph=True)[0]
        p_rr_im = \
            autograd.grad(p_r_im.view(-1, 1), g, torch.ones(input.view(-1, 1).shape).to(self.device),
                          create_graph=True)[0]
        p_xx = p_rr_re[:, [0]] + 1j* p_rr_im[:, [0]]
        p_yy = p_rr_re[:, [1]] + 1j* p_rr_im[:, [1]]

        # given that x, y are scaled here so that x' = x/c and y' = y/c, then c = 1
        # f = p_tt - self.c * (p_xx + p_yy)
        f = p_xx + p_yy + k**2 * torch.view_as_complex(pnet)

        loss_f = self.loss_function_pde(f.view(-1, 1), torch.zeros_like(f.view(-1, 1)))

        return loss_f

    def loss_bc(self, input):
        g = input.clone()
        g = self.scale_f(g)
        g.requires_grad = True
        k = 2*np.pi*g[:, 2]/self.c
        r, phi = self.cylindrical_coords(input)
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)

        # if self.ansatz_formulation:
        #     pressure = (self.dnn(g)*plane_wave(g[0,2],
        #                                        1.,
        #                                        self.fs/self.c,
        #                                        self.r_source,
        #                                        g[0, :2])).sum(axis = -1).unsqueeze(0)
        # else:
        pnet = self.dnn(g)

        # pnet = self.output_scaler.backward_rir(pressure)
        p_r_re = autograd.grad(pnet[0], g, torch.ones_like(pnet[0]),
                                create_graph=True)[0]
        p_r_im = autograd.grad(pnet[1], g, torch.ones_like(pnet[1]),
                                create_graph=True)[0]

        p_x = p_r_re[:, 0] + 1j*p_r_im[:, 0]
        p_y = p_r_re[:, 1] + 1j*p_r_im[:, 1]
        dp_dr = sin_phi * p_y + cos_phi * p_x
        # Sommerfeld radiation condition (eq. 4.5.5 - "Acoustics" - Allan D. Pierce)
        # f = r * (dp_dr + 1 / self.c * dp_dt)
        # f = r * (dp_dr + dp_dt)
        f = dp_dr -1j*k*torch.view_as_complex(pnet)
        bcs_loss = self.loss_function_pde(f.view(-1, 1), torch.zeros_like(f.view(-1, 1)))
        return bcs_loss

    def loss_ic(self, input):
        # x,y,t = input
        g = input.clone()
        g.requires_grad = True

        pnet = self.dnn(g).T
        k = 2*np.pi*g[:, 2]/self.c
        r, phi = self.cylindrical_coords(input)
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)

        # pnet = self.output_scaler.backward_rir(pressure)

        p_r_re = autograd.grad(pnet[0], g, torch.ones_like(pnet[0]),
                                create_graph=True)[0]
        p_r_im = autograd.grad(pnet[1], g, torch.ones_like(pnet[1]),
                                create_graph=True)[0]
        p_x = p_r_re[:, 0] + 1j*p_r_im[:, 0]
        p_y = p_r_re[:, 1] + 1j*p_r_im[:, 1]
        dp_dr = sin_phi * p_y + cos_phi * p_x
        f = dp_dr
        ics_loss = self.loss_function_pde(f.view(-1, 1), torch.zeros_like(f.view(-1, 1)))
        return ics_loss

    def loss(self, inpuf_data, inpuf_pde, inpuf_ic, pm, data_loss_weights):
        if self.output_scaler is not None:
            pm = self.output_scaler.forward_rir(pm)
        loss_p, norm_ratio, std_ratio, maxabs_ratio = self.loss_data(inpuf_data, pm, data_loss_weights)
        loss_f = self.loss_PDE(inpuf_pde)
        loss_bc = self.loss_bc(inpuf_pde)
        loss_ic = self.loss_ic(inpuf_ic)

        loss_val = self.lambda_data * loss_p + self.lambda_pde * loss_f + self.lambda_bc * loss_bc \
                   + self.lambda_ic * loss_ic

        return loss_val, loss_p, loss_f, loss_bc, loss_ic, norm_ratio, std_ratio, maxabs_ratio

    'callable for optimizer'

    def closure(self, optimizer, train_input, test_input, p_train, p_test):

        optimizer.zero_grad()

        loss = self.loss(train_input, p_train)

        loss.backward()

        self.iter += 1

        if self.iter % 100 == 0:
            error_vec, _, _ = self.fest(test_input, p_test)
            # TODO: FIX HERE
            print(
                'Relative Error (Test): %.5f' %
                (
                    error_vec.cpu().detach().numpy(),
                )
            )

        return loss

    def SGD_step(self, data_input, pde_input, ic_input, p_data, data_loss_weights = None):

        loss, loss_data, loss_pde, loss_bc, loss_ic, norm_ratio, std_ratio, maxabs_ratio \
            = self.loss(data_input, pde_input, ic_input, p_data, data_loss_weights)

        loss.backward()

        # self.iter += 1
        #
        # if self.iter % 100 == 0:
        #     error_vec, relative_r _ = self.fest(test_input, p_test)
        #     # TODO: FIX HERE
        #     print(
        #         'Relative Error (Test): %.5f' %
        #         (
        #             error_vec.cpu().detach().numpy(),
        #         )
        #     )

        return (loss.cpu().detach().numpy(),
                loss_data.cpu().detach().numpy(),
                loss_pde.cpu().detach().numpy(),
                loss_bc.cpu().detach().numpy(),
                loss_ic.cpu().detach().numpy(),
                norm_ratio.cpu().detach().numpy(),
                std_ratio.cpu().detach().numpy(),
                maxabs_ratio.cpu().detach().numpy()
                )

    'test neural network'

    def test(self, test_input, p_true):
        g = test_input.clone().unsqueeze(0)
        g = self.scale_f(g)

        p_pred = self.dnn(g)
        p_pred = torch.view_as_complex(p_pred)
        if self.output_scaler is not None:
            p_pred = self.output_scaler.backward_rir(p_pred)
        # Relative L2 Norm of the error
        # relative_error = torch.linalg.norm((p_true - p_pred.squeeze(0)), 2) / torch.linalg.norm(p_true,2)
        sq_err = torch.abs(p_true.to(self.device) - p_pred) ** 2
        mse = sq_err.mean()
        # Error vector
        error_vec = torch.abs((p_true.to(self.device) - p_pred)**2 / (p_true.to(self.device) + np.finfo(np.float32).eps)**2)
        p_pred = p_pred.squeeze(0).cpu().detach().numpy()
        error_vec = error_vec.squeeze(0).cpu().detach().numpy()

        return error_vec, mse.item(), sq_err.cpu().detach().numpy(), p_pred

    def inference(self, input):
        g = input.clone()
        g = self.scale_f(g)
        p_pred = self.dnn(g).T
        # if self.output_scaler is not None:
        #     p_pred = self.output_scaler.backward_rir(p_pred)
        return p_pred

    def scale_xy(self, input):
        x, y, f = input[:, 0, :].flatten(), input[:, 1, :].flatten(), input[:, 2, :].flatten()
        x = x / self.c
        y = y / self.c
        return torch.stack((x, y, f), dim=-1).view(3, -1).unsqueeze(0)

    def scale_f(self, input):
        x, y, f = input[:, 0, :].flatten(), input[:, 1, :].flatten(), input[:, 2, :].flatten()
        f = f / self.c
        return torch.vstack((x, y, f)).unsqueeze(0)


class PINNDataset(Dataset):
    def __init__(self,
                 refdata,
                 measured_data,
                 x_ref,
                 y_ref,
                 x_m,
                 y_m,
                 f,
                 f_ind,
                 n_pde_samples=800,
                 counter=1,
                 maxcounter=1e5,
                 curriculum_training=False,
                 batch_size = 300):
        self.tfnp = lambda x: torch.from_numpy(x).float()
        self.tfnp_cmplx = lambda x: torch.from_numpy(x).type(torch.complex64)
        self.curriculum_training = curriculum_training
        self.counter = counter
        self.maxcounter = maxcounter
        # self.maxcounter = -1
        self.trainData = measured_data
        self.n_pde_samples = n_pde_samples
        # self.BCData = refdata[x_y_boundary_ind]
        self.f_ind = f_ind
        self.batch_size = batch_size
        # self.x_y_boundary_ind = x_y_boundary_ind
        self.x_m = x_m
        self.y_m = y_m
        self.x_ref = x_ref
        self.y_ref = y_ref
        self.f = f
        self.ff = np.repeat(self.f, len(self.x_ref))
        self.xx = np.tile(self.x_ref, len(self.f))
        self.yy = np.tile(self.y_ref, len(self.f))
        self.collocation_all = self.tfnp(np.stack([self.xx, self.yy, self.ff], axis=0))
        self.pressure_all = self.tfnp_cmplx(refdata[:, self.f_ind].flatten())
        self.xmax = self.x_ref.max()
        self.xmin = self.x_ref.min()
        self.ymax = self.y_ref.max()
        self.ymin = self.y_ref.min()
        self.fmax = self.f[self.f_ind].max()
        self.counter_fun = lambda x, n: int(n * x)
        decay_rate = np.linspace(0, 1, len(self.f))
        self.f_weight = 10*(1-.98)**decay_rate

    def __len__(self):
        return 1
        # return len(self.f_ind)

    def __getitem__(self, idx):
        if np.logical_and(self.curriculum_training, self.counter < self.maxcounter):
            sample_limit = self.counter_fun(self.counter / self.maxcounter, len(self.f_ind))
            sample_limit = np.maximum(self.batch_size, sample_limit)
            idx = np.random.randint(0, sample_limit, self.batch_size)
            f_batch_indx = self.f_ind[idx]
            f_ind_temp = self.f_ind[:sample_limit]
            f_lims = (self.f[f_ind_temp].min(), self.f[f_ind_temp].max())
        elif np.logical_and(not self.curriculum_training, self.counter < self.maxcounter):
            window_size = self.batch_size
            overlap = self.batch_size // 2
            f_ind_windowed = window(self.f_ind, w=window_size, o=overlap)  # 100 taps, 25 overlap
            n_windows = f_ind_windowed.shape[0]
            window_number = self.counter_fun(self.counter / self.maxcounter, n_windows)
            # f_ind_temp = self.f_ind[:(progressive_t_counter + 1)]
            f_ind_temp = f_ind_windowed[window_number]
            # idx = np.random.randint(0, progressive_t_counter + 1)
            idx = np.random.randint(0, window_size, window_size)
            f_batch_indx = f_ind_temp[idx]
            f_lims = (self.f[f_ind_temp].min(), self.f[f_ind_temp].max())
        else:
            idx = np.random.randint(0, len(self.f_ind), self.batch_size)
            f_batch_indx = self.f_ind[idx]
            f_lims = (self.f[self.f_ind].min(), self.f[self.f_ind].max())
        f_data = self.f[f_batch_indx]
        pressure_batch = self.trainData[:, f_batch_indx].flatten(order = 'F')
        pressure_bc_batch = self.trainData[:, f_batch_indx].flatten(order = 'F')
        x_data, y_data = self.x_m, self.y_m

        grid_pde = (2 * (lhs(2, self.n_pde_samples)) / 1 - 1)
        grid_ic = (2 * (lhs(2, self.n_pde_samples)) / 1 - 1)
        x_pde = self.xmax * grid_pde[:, 0]
        y_pde = self.ymax * grid_pde[:, 1]
        x_bc = self.xmax * grid_pde[:, 0]
        y_bc = self.ymax * grid_pde[:, 1]
        x_ic = self.xmax * grid_ic[:, 0]
        y_ic = self.ymax * grid_ic[:, 1]
        f_ic = np.zeros(self.n_pde_samples)
        if self.counter < self.maxcounter:
            f_pde = f_data.max() * lhs(1, self.n_pde_samples).squeeze(-1)
            f_bc = f_data.max() * lhs(1, self.n_pde_samples).squeeze(-1)
        else:
            f_pde = self.fmax * lhs(1, self.n_pde_samples).squeeze(-1)
            f_bc = f_data.max() * lhs(1, self.n_pde_samples).squeeze(-1)

        data_loss_weights = self.f_weight[f_batch_indx]
        data_loss_weights = np.repeat(data_loss_weights, len(x_data))
        tf_data = np.repeat(f_data, len(x_data))
        xx_data = np.tile(x_data, len(f_data))
        yy_data = np.tile(y_data, len(f_data))
        collocation_train = np.stack([xx_data, yy_data, tf_data], axis=0)
        collocation_pde = np.stack([x_pde, y_pde, f_pde], axis=0)
        collocation_bc = np.stack([x_bc, y_bc, f_bc], axis=0)
        collocation_ic = np.stack([x_ic, y_ic, f_ic], axis=0)
        self.counter += 1

        return {
            'collocation_train': self.tfnp(collocation_train),
            'collocation_bc': self.tfnp(collocation_bc),
            'collocation_pde': self.tfnp(collocation_pde),
            'collocation_ic': self.tfnp(collocation_ic),
            'pressure_bc_batch': self.tfnp_cmplx(pressure_bc_batch),
            'pressure_batch': self.tfnp_cmplx(pressure_batch),
            'f_batch_indx': f_batch_indx,
            'max_f': f_data.max(),
            'data_loss_weights': self.tfnp(data_loss_weights),
            'f_lims': f_lims,}


def window(a, w=4, o=2, copy=False):
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides=st, shape=sh)[0::o]
    if copy:
        return view.copy()
    else:
        return view
