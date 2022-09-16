import sys
sys.path.append('../')
import torch
import torch.autograd as autograd  # computation graph
# from torch import Tensor  # tensor node in the computation graph
import torch.nn as nn  # neural networks
import numpy as np
from torch.utils.data import Dataset
from utils_soundfields import plot_sf, plot_array_pressure
# from tueplots import axes, bundles
import matplotlib.pyplot as plt
import os
import glob
import re
from SIREN import Siren, SirenNet
# from modules_meta import SingleBVPNet

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

def construct_input_vec(rirdata, x_true, y_true, t, data_ind = None, t_ind = None):
    if data_ind is not None:
        rirdata = rirdata[data_ind]
        x_true = x_true[data_ind]
        y_true = y_true[data_ind]
    if t_ind is not None:
        rirdata = rirdata[:, t_ind]
        t = t[t_ind]
    collocation = []
    for i in range(len(t)):
        tt = np.repeat(t[i], len(x_true))
        collocation.append(np.stack([x_true, y_true, tt], axis=0))
    return np.array(collocation), rirdata

def plot_results(collocation_data, rirdata, PINN):
    Nplots = collocation_data.shape[0]
    Pred_pressure = []
    error_vecs = []
    relative_errors = []
    tfnp = lambda x: torch.from_numpy(x).float().to(PINN.device)

    for n in range(Nplots):
        error_vec, relative_error, p_pred = PINN.test(tfnp(collocation_data[n]), tfnp(rirdata[:,n]))
        Pred_pressure.append(p_pred.squeeze(0))
        error_vecs.append(error_vec)
        relative_errors.append(relative_error)
    fig, axes = plt.subplots(nrows=3, ncols=Nplots, sharex=True, sharey=True)
    error_vec_minmax = (np.array(error_vecs).min(), np.array(error_vecs).min()+ 1.)
    p_pred_minmax = (np.array(Pred_pressure).min(), np.array(Pred_pressure).max())
    p_minmax = (rirdata.min(), rirdata.max())
    for i, ax in enumerate(axes[0]):
        if i == 2:
            name = 'Predicted - \n'
        else:
            name = ''
        ax, im = plot_array_pressure(Pred_pressure[i], collocation_data[0, 0:2], ax=ax, norm = p_pred_minmax)
        ax.set_title(name +'t = {:.2f}s'.format(collocation_data[i, 2, 0]))
        # ax, im = plot_sf(Pred_pressure[i], collocation_data[i, 0], collocation_data[i, 1],
        #                 ax=ax, name= name +'t = {:.2f}s'.format(collocation_data[i, 2, 0]),
        #                 clim= p_pred_minmax)
        if i != 0:
            ax.set_ylabel('')
    for i, ax in enumerate(axes[1]):
        if i == 2:
            name = 'True'
        else:
            name = ''
        ax, im2 = plot_array_pressure(rirdata[:,i], collocation_data[0, 0:2], ax=ax, norm = p_minmax)
        ax.set_title(name)
        # ax, _ = plot_sf(rirdata[:,i], collocation_data[i, 0], collocation_data[i, 1],
        #                 ax=ax, name = name,
        #                 clim = p_pred_minmax)
        if i != 0:
            ax.set_ylabel('')
    for i, ax in enumerate(axes[2]):
        if i == 2:
            name = 'Relative Error'
        else:
            name = ''
        ax, im3 = plot_array_pressure(error_vecs[i], collocation_data[0, 0:2], ax=ax,
                                      norm=error_vec_minmax, cmp = 'hot')
        ax.set_title(name)
        # ax, im2 = plot_sf(error_vecs[i], collocation_data[i, 0], collocation_data[i, 1],
        #                 ax=ax, name = name, clim = error_vec_minmax, cmp = 'hot')
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


    return fig, np.array(relative_errors)

#  Deep Neural Network
class DNN(nn.Module):
    def __init__(self, layers, lb, ub, siren = True):
        super().__init__()  # call __init__ from parent class
        self.siren = siren
        'activation function'
        if self.siren:
            self.activation = nn.Identity()
        else:
            self.activation = nn.Tanh()

        self.lb = lb
        self.ub = ub
        self.layers = layers
        self.scaling = lambda x: 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0

        'Initialize neural network as a list using nn.Modulelist'
        if self.siren:
            # self.linears = nn.ModuleList([Siren(dim_in=layers[i],dim_out=layers[i + 1]) for i in range(len(layers) - 1)])
            self.net = SirenNet(
                            dim_in = layers[0],               # input dimension, ex. 2d coor
                            dim_hidden = 256,                 # hidden dimension
                            dim_out = 1,                      # output dimension, ex. rgb value
                            num_layers = 5,         # number of layers
                            final_activation = nn.Identity(), # activation of final layer (nn.Identity() for direct output)
                            w0_initial = 30.,                  # different signals may require different omega_0 in the first layer - this is a hyperparameter
                            w0 = 1.
                        )
            # self.net = SingleBVPNet(out_features= 1, in_features= 3, hidden_features= 256, num_hidden_layers= 4)
        else:
            self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
            'Xavier Normal Initialization'
            for i in range(len(layers) - 1):
                nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)

                # set biases to zero
                nn.init.zeros_(self.linears[i].bias.data)

    'foward pass'
    def forward(self, input):
        batch_size = input.shape[0]
        g = input.clone()
        x, y, t = g[:, 0, :].flatten(), g[:, 1, :].flatten(), g[:, 2, :].flatten()

        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)
        if torch.is_tensor(y) != True:
            y = torch.from_numpy(y)
        if torch.is_tensor(t) != True:
            t = torch.from_numpy(t)

        # convert to float
        x = x.float()
        y = y.float()
        t = t.float()

        # preprocessing input - feature scaling function
        input_preprocessed = torch.stack((x, y, t), dim=-1).view(-1, 3)
        z = self.scaling(input_preprocessed)
        if self.siren:
            p_out = self.net(z) # z: [batchsize x 3]
        else:
            for i in range(len(self.layers) - 2):
                z = self.linears[i](z)
                z = self.activation(z)

            p_out = self.linears[-1](z)

        return p_out.reshape(batch_size, -1)


#  PINN
# https://github.com/alexpapados/Physics-Informed-Deep-Learning-Solid-and-Fluid-Mechanics
class FCN():
    def __init__(self,
                 layers,
                 bounds,
                 device=None,
                 siren = True,
                 lambda_data = 1.,
                 lambda_pde = 1e-4,
                 lambda_bc = 1e-2):
        if device is None:
            self.device = 'cpu'
        else:
            self.device = device
        # self.loss_function = nn.MSELoss(reduction='mean')
        # self.loss_function = nn.MSELoss(reduction='sum')
        self.loss_function = nn.L1Loss(reduction='mean')
        'Initialize iterator'
        self.iter = 0
        self.lambda_data = lambda_data
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.siren = siren
        'speed of sound'

        (self.xmin, self.xmax) = bounds['x']
        (self.ymin, self.ymax) = bounds['y']
        (self.tmin, self.tmax) = bounds['t']
        self.c = 343. / max(self.xmax, self.ymax) * self.tmax

        self.lb = torch.Tensor([self.xmin, self.ymin, self.tmin]).to(self.device)
        self.ub = torch.Tensor([self.xmax, self.ymax, self.tmax]).to(self.device)
        'Call our DNN'
        self.dnn = DNN(layers, lb=self.lb, ub= self.ub, siren = siren).to(device)

    def cylindrical_coords(self, input):
        x, y, t = input[:, 0, :].flatten(), input[:, 1, :].flatten(), input[:, 2, :].flatten()
        r = torch.sqrt(x**2 + y**2)
        phi = torch.atan2(y, x)
        return r, phi

    def loss_data(self, input, pm):

        loss_u = self.loss_function(self.dnn(input), pm)

        return loss_u

    def loss_PDE(self, input):

        g = input.clone()
        gscaled = input.clone()
        gscaled = self.scale_xy(gscaled)
        g.requires_grad = True
        gscaled.requires_grad = True

        pnet = self.dnn(g)
        pnetscaled = self.dnn(gscaled)

        p_r_t = \
            autograd.grad(pnet.view(-1, 1), g, torch.ones([input.view(-1, 3).shape[0], 1]).to(self.device),
                          retain_graph=True,
                          create_graph=True)[0]
        p_rr_tt = \
            autograd.grad(p_r_t.view(-1, 1), g, torch.ones(input.view(-1, 1).shape).to(self.device),
                          create_graph=True)[0]
        p_r_t_scaled = \
            autograd.grad(pnetscaled.view(-1, 1), gscaled, torch.ones([input.view(-1, 3).shape[0], 1]).to(self.device),
                          retain_graph=True,
                          create_graph=True)[0]
        p_rr_tt_scaled = \
            autograd.grad(p_r_t_scaled.view(-1, 1), gscaled, torch.ones(input.view(-1, 1).shape).to(self.device),
                          create_graph=True)[0]

        p_xx = p_rr_tt_scaled[:, [0]]
        p_yy = p_rr_tt_scaled[:, [1]]
        p_tt = p_rr_tt[:, [2]]

        # given that x, y are scaled here so that x' = x/c and y' = y/c, then c = 1
        # f = p_tt - self.c * (p_xx + p_yy)
        f = p_tt - 1.*(p_xx + p_yy)

        loss_f = self.loss_function(f.view(-1,1), torch.zeros_like(f.view(-1,1)))

        return loss_f

    def loss_bc(self, input):
        # x,y,t = input
        g = input.clone()
        g.requires_grad = True

        gscaled = input.clone()
        gscaled = self.scale_xy(gscaled)
        gscaled.requires_grad = True

        r, phi = self.cylindrical_coords(gscaled)

        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)
        pnet = self.dnn(g)
        pnetscaled = self.dnn(gscaled)

        p_x_y_t = autograd.grad(pnet.view(-1, 1), g, torch.ones([input.view(-1, 3).shape[0], 1]).to(self.device),
                          create_graph=True)[0]
        p_x_y_t_scaled = autograd.grad(pnetscaled.view(-1, 1), gscaled,
                                       torch.ones([input.view(-1, 3).shape[0], 1]).to(self.device),
                                       create_graph=True)[0]
        p_x = p_x_y_t_scaled[:, [0]].flatten()
        p_y = p_x_y_t_scaled[:, [1]].flatten()
        dp_dt = p_x_y_t[:, [2]].flatten()
        dp_dr = sin_phi * p_y + cos_phi * p_x
        # Sommerfeld conditions
        # given that x, y are scaled here so that x' = x/c and y' = y/c, then c = 1
        # f = r * (dp_dr + 1 / self.c * dp_dt)
        f = r * (dp_dr + 1. * dp_dt)
        bcs_loss = self.loss_function(f.view(-1,1), torch.zeros_like(f.view(-1,1)))
        return bcs_loss

    def loss(self, input_data, input_pde, pm):

        loss_p = self.loss_data(input_data, pm)
        loss_f = self.loss_PDE(input_pde)
        loss_bc = self.loss_bc(input_pde)

        # loss_val = 1e2*loss_p + 1e-3*loss_f
        if self.siren:
            loss_val = self.lambda_data*loss_p + self.lambda_pde*loss_f + self.lambda_bc*loss_bc
        else:
            loss_val = loss_p + (loss_f + loss_bc)

        return loss_val, loss_p, loss_f, loss_bc

    'callable for optimizer'

    def closure(self, optimizer, train_input, test_input, p_train, p_test):

        optimizer.zero_grad()

        loss = self.loss(train_input, p_train)

        loss.backward()

        self.iter += 1

        if self.iter % 100 == 0:
            error_vec, _, _ = self.test(test_input, p_test)
            # TODO: FIX HERE
            print(
                'Relative Error (Test): %.5f' %
                (
                    error_vec.cpu().detach().numpy(),
                )
            )

        return loss

    def SGD_step(self, data_input, pde_input, p_data):

        loss, loss_data, loss_pde, loss_bc = self.loss(data_input, pde_input, p_data)

        loss.backward()

        # self.iter += 1
        #
        # if self.iter % 100 == 0:
        #     error_vec, relative_r _ = self.test(test_input, p_test)
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
                loss_bc.cpu().detach().numpy())

    'test neural network'

    def test(self, test_input, p_true):

        p_pred = self.dnn(test_input.unsqueeze(0))
        # Relative L2 Norm of the error
        # relative_error = torch.linalg.norm((p_true - p_pred.squeeze(0)), 2) / torch.linalg.norm(p_true,2)
        relative_error = (torch.abs(p_true - p_pred.squeeze(0))**2).mean()
        # Error vector
        error_vec = torch.abs(p_true - p_pred.squeeze(0) / (p_true + np.finfo(np.float32).eps))
        p_pred = p_pred.cpu().detach().numpy()
        error_vec = error_vec.cpu().detach().numpy()

        return error_vec, relative_error.item(), p_pred

    def scale_xy(self, input):
        x, y, t = input[:, 0, :].flatten(), input[:, 1, :].flatten(), input[:, 2, :].flatten()
        x = x/self.c
        y = y/self.c
        return torch.stack((x, y, t), dim=-1).view(3, -1).unsqueeze(0)


class PINNDataset(Dataset):
    def __init__(self,
                 rirdata,
                 x_true,
                 y_true,
                 t,
                 data_ind,
                 x_y_boundary_ind,
                 t_ind,
                 device = 'cuda',
                 n_pde_samples = 800):
        self.tfnp = lambda x : torch.from_numpy(x).float()
        self.counter = 1
        # self.maxcounter = int(1e9)
        self.maxcounter = -1
        self.TrainData = rirdata[data_ind]
        self.n_pde_samples = n_pde_samples
        self.BCData = rirdata[x_y_boundary_ind.squeeze(-1)]
        self.t_ind = t_ind
        self.data_ind = data_ind
        self.x_y_boundary_ind = x_y_boundary_ind.squeeze(-1)
        self.x_true = x_true
        self.y_true = y_true
        self.t = t
        self.tt = np.repeat(self.t, len(self.x_true))
        self.xx = np.tile(self.x_true, len(self.t))
        self.yy = np.tile(self.y_true, len(self.t))
        self.collocation_all = self.tfnp(np.stack([self.xx, self.yy, self.tt], axis = 0))
        self.pressure_all = self.tfnp(rirdata[:, self.t_ind].flatten())
        self.xmax = self.x_true.max()
        self.xmin = self.x_true.min()
        self.ymax = self.y_true.max()
        self.ymin = self.y_true.min()
        self.tmax = self.t[self.t_ind].max()
        # self.batch_size = batch_size
        # self.n_time_instances = int(0.6 * self.batch_size)
        # self.n_spatial_instances = self.batch_size - self.n_time_instances
        # self.n_spatial_instances = len(data_ind)
        # self.n_time_instances = self.batch_size - self.n_spatial_instances

    def __len__(self):
        return 1
        # return len(self.t_ind)

    def __getitem__(self, idx):
        if self.counter < self.maxcounter:
            progressive_t_counter = max(self.counter//1000, 100)
            progressive_t_counter = min(progressive_t_counter, len(self.t_ind))
            t_ind_temp = self.t_ind[self.t_ind < progressive_t_counter]
            idx = np.random.randint(0, progressive_t_counter)
            t_batch_indx = t_ind_temp[idx]
        else:
            idx = np.random.randint(0, len(self.t_ind))
            t_batch_indx = self.t_ind[idx]
        t_data = self.t[t_batch_indx]
        pressure_batch = self.TrainData[:, t_batch_indx].flatten()
        pressure_bc_batch = self.BCData[:, t_batch_indx].flatten()
        x_data, y_data = self.x_true[self.data_ind], self.y_true[self.data_ind]
        x_pde = torch.FloatTensor(self.n_pde_samples).uniform_(self.xmin, self.xmax)
        y_pde = torch.FloatTensor(self.n_pde_samples).uniform_(self.ymin, self.ymax)
        x_bc = torch.FloatTensor(self.n_pde_samples).uniform_(self.xmin, self.xmax)
        y_bc = torch.FloatTensor(self.n_pde_samples).uniform_(self.ymin, self.ymax)
        if self.counter < self.maxcounter:
            t_pde = torch.FloatTensor(self.n_pde_samples).uniform_(0., t_data.max())
            t_bc = torch.FloatTensor(self.n_pde_samples).uniform_(0., t_data.max())
        else:
            t_pde = torch.FloatTensor(self.n_pde_samples).uniform_(0., self.tmax)
            t_bc = torch.FloatTensor(self.n_pde_samples).uniform_(0., self.tmax)
        tt_data = np.repeat(t_data, len(x_data))
        collocation_train = np.stack([x_data, y_data, tt_data], axis = 0)
        collocation_pde = np.stack([x_pde, y_pde, t_pde], axis = 0)
        collocation_bc = np.stack([x_bc, y_bc, t_bc], axis = 0)
        self.counter += 1

        return {'collocation_train' : self.tfnp(collocation_train),
                'collocation_bc' : self.tfnp(collocation_bc),
                'collocation_pde' : self.tfnp(collocation_pde),
                'pressure_bc_batch' : self.tfnp(pressure_bc_batch),
                'pressure_batch' : self.tfnp(pressure_batch),
                't_batch_indx' : t_batch_indx,
                'max_t': t_data.max()}

# class WaveEqDataset(Dataset):
#     class SingleHelmholtzSource(Dataset):
#         def __init__(self,
#                      rirdata,
#                      x_true,
#                      y_true,
#                      t,
#                      data_ind,
#                      x_y_boundary_ind,
#                      t_ind,
#                      nsamples = 5000,
#                      device = 'cuda'):
#             super().__init__()
#             torch.manual_seed(0)
#             self.tfnp = lambda x : torch.from_numpy(x).float().to(device)
#             self.counter = 0
#             self.full_count = 100e3
#             self.TrainData = rirdata[data_ind]
#             self.BCData = rirdata[x_y_boundary_ind.squeeze(-1)]
#             self.t_ind = t_ind
#             self.data_ind = data_ind
#             self.x_y_boundary_ind = x_y_boundary_ind.squeeze(-1)
#             self.x_true = x_true
#             self.y_true = y_true
#             self.nsamples = nsamples
#             self.t = t
#             self.tt = np.repeat(self.t, len(self.x_true))
#             self.xx = np.tile(self.x_true, len(self.t))
#             self.yy = np.tile(self.y_true, len(self.t))
#             self.collocation_all = self.tfnp(np.stack([self.xx, self.yy, self.tt], axis = 0))
#             self.pressure_all = self.tfnp(rirdata[:, self.t_ind].flatten())
#             self.xmax = self.x_true.max() + 0.01*self.x_true.max()
#             self.xmin = self.x_true.min() + 0.01*self.x_true.min()
#             self.rmax = 1.6
#             self.ymax = self.y_true.max() + 0.01*self.y_true.max()
#             self.ymin = self.y_true.min() + 0.01*self.y_true.min()
#             self.tmax = self.t[self.t_ind].max() + 0.1*self.t[self.t_ind].max()
#
#         def __len__(self):
#             return 1
#
#         def __getitem__(self, idx):
#             # indicate where border values are
#             t_batch_indx = self.t_ind[np.random.randint(0, int(len(self.t_ind)*(self.counter / self.full_count)))]
#             t_data = self.t[t_batch_indx]
#             pressure_batch = self.TrainData[:, t_batch_indx].flatten()
#             pressure_bc_batch = self.BCData[:, t_batch_indx].flatten()
#             x_data, y_data = self.x_true[self.data_ind], self.y_true[self.data_ind]
#             tt_data = np.repeat(t_data, len(x_data))
#             collocation_train = np.stack([x_data, y_data, tt_data], axis=0)
#             N_train_data = len(collocation_train)
#             # random coordinates
#             length = torch.sqrt(torch.FloatTensor(self.nsamples,).uniform_(0., self.rmax**2))
#             angle = np.pi *  torch.FloatTensor(self.nsamples,).uniform_(0., 2.)
#             x = length * torch.cos(angle)
#             y = length * torch.sin(angle)
#             coords = torch.concat((x[..., None], y[..., None]), axis = -1)
#
#             time = torch.zeros(self.nsamples, 1).uniform_(0, 0.4 * (self.counter / self.full_count))
#             coords = torch.cat([coords, time], axis = -1)
#             # make sure we always have training samples from data
#             coords[-N_train_data] = torch.cat([coords, time], axis = -1)
#
#
#
#             return {'coords': coords}, {'source_boundary_values': boundary_values,
#                                         'gt': self.field,
#                                         'sound_speed': c,
#                                         'sound_speed_grid': c_grid,
#                                         'mgrid': self.mgrid,
#                                         'coordinate_grid': coords,
#                                         'wavenumber': self.wavenumber,
#                                         'omega': self.omega,
#                                         'sidelength': self.sidelength,
#                                         'samples': self.samples,
#                                         'c_scale': self.c_scale,
#                                         'ground_impedance': self.ground_impedance,
#                                         'source_indices': source_indices,
#                                         'bc_indx_dict': bc_indx_dict,
#                                         'bc_indices': bc_indices}
