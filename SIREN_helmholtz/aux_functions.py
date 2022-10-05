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
# from SIREN import Siren, SirenNet
from modules_meta import SingleBVPNet

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
    p_pred_minmax = (np.array(Pred_pressure).min(),(np.array(Pred_pressure).max()+ rirdata.max())/2)
    p_true_minmax = (rirdata.min(), (np.array(Pred_pressure).max()+ rirdata.max())/2)
    for i, ax in enumerate(axes[0]):
        if i == 2:
            name = 'Predicted - \n'
        else:
            name = ''
        ax, im = plot_sf(Pred_pressure[i], collocation_data[i, 0], collocation_data[i, 1],
                        ax=ax, name= name +'t = {:.2f}s'.format(collocation_data[i, 2, 0]),
                        clim= p_pred_minmax)
        if i != 0:
            ax.set_ylabel('')
    for i, ax in enumerate(axes[1]):
        if i == 2:
            name = 'True'
        else:
            name = ''
        ax, im2 = plot_sf(rirdata[:,i], collocation_data[i, 0], collocation_data[i, 1],
                        ax=ax, name = name,
                        clim = p_true_minmax)
        if i != 0:
            ax.set_ylabel('')
    for i, ax in enumerate(axes[2]):
        if i == 2:
            name = 'Relative Error'
        else:
            name = ''
        ax, im3 = plot_sf(error_vecs[i], collocation_data[i, 0], collocation_data[i, 1],
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
            # self.net = SirenNet(
            #                 dim_in = layers[0],               # input dimension, ex. 2d coor
            #                 dim_hidden = 256,                 # hidden dimension
            #                 dim_out = 1,                      # output dimension, ex. rgb value
            #                 num_layers = 5,         # number of layers
            #                 final_activation = nn.Identity(), # activation of final layer (nn.Identity() for direct output)
            #                 w0_initial = 30.,                  # different signals may require different omega_0 in the first layer - this is a hyperparameter
            #                 w0 = 1.
            #             )
            self.net = SingleBVPNet(out_features= 2, in_features= 3, hidden_features= 512, num_hidden_layers= 3)
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

class scale_net_out:
    """
    scale 1 or 2 dimensional data
    Usage
    ------------------------------
    scaler = scale_net_out(c)
    c_scaled = scaler.scale_data(c)
    """

    def __init__(self, c_dummy_vector):
        self.ndim = c_dummy_vector.ndim
        self.dummy = c_dummy_vector
        self.p2p = np.ptp(c_dummy_vector.cpu())
        self.mindata = torch.min(c_dummy_vector)

    def scale_data(self, data):
        if self.ndim < 2:
            d = data
            return 2. * (d - self.mindata) / self.p2p - 1
        else:
            d = data[..., 0]
            scaled = 2. * (d - self.mindata) / self.p2p - 1
            return torch.stack((scaled, data[..., 1]), axis=-1)

    def unscale_data(self, data):
        if data.ndim > 1:
            d = data[..., 0]
        else:
            d = data
        unscaled = (d + 1) * self.p2p / 2. + self.mindata
        return torch.stack((unscaled, torch.zeros_like(d)), axis=-1)
#  PINN
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
        self.c = 343.
        # self.c = 343. / max(self.xmax, self.ymax) * self.tmax

        (self.xmin, self.xmax) = bounds['x']
        (self.ymin, self.ymax) = bounds['y']
        (self.tmin, self.tmax) = bounds['t']

        # self.xmin /=self.c
        # self.xmax /=self.c
        # self.ymin /=self.c
        # self.ymax /=self.c
        self.tmax *=self.c
        self.tmin *=self.c

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
        g = input.clone()
        g = self.scale_t(g)
        g.requires_grad = True

        loss_u = self.loss_function(self.dnn(g), pm)

        return loss_u

    def loss_helmholtz(self, model_output, gt):
        source_boundary_values = gt['source_boundary_values']

        if 'rec_boundary_values' in gt:
            rec_boundary_values = gt['rec_boundary_values']

        omega = gt['omega'].float()
        x = model_output['model_in']  # (meta_batch_size, num_points, 2)
        y = model_output['model_out']  # (meta_batch_size, num_points, 2)
        c = gt['sound_speed']
        batch_size = x.shape[1]
        scaler = scale_net_out(gt['c_scale'])

        if 'fit_c' in gt:
            pred_c = scaler.unscale_data(y[0, :, -1])
            if gt['fit_c']:
                full_waveform_inversion = True
                c_pml = scaler.unscale_data(torch.zeros_like(y[0, :, -1]))  # dummy variable
                c = torch.where((torch.abs(x[..., 0, None]) > 0.75) | (torch.abs(x[..., 1, None]) > 0.75),
                                c_pml, c)
            y = y[:, :, :-1]

        dp, status = diff_operators.jacobian(y, x)
        dpdx1 = dp[..., 0]
        dpdx2 = dp[..., 1]

        a0 = 5.
        rho = 1.2
        # let pml extend from -1. to -1 + Lpml and 1 - Lpml to 1.0
        Lpml = 0.5
        dist_west = -torch.clamp(x[..., 0] + (1.0 - Lpml), max=0)
        dist_east = torch.clamp(x[..., 0] - (1.0 - Lpml), min=0)
        dist_north = torch.clamp(x[..., 1] - (1.0 - Lpml), min=0)

        if torch.all(gt['ground_impedance'].isnan()).item():
            dist_south = -torch.clamp(x[..., 1] + (1.0 - Lpml), max=0)
            bc_mask = torch.zeros_like(y).to(torch.bool)
            diff_bc = torch.zeros_like(y)
        else:
            dist_south = torch.zeros_like(dist_north)
            bc_mask = torch.ones_like(y).to(torch.bool)
            bc_mask[..., 1] = torch.where(x[..., 1] == x[..., 1].min(),
                                          torch.ones_like(x[..., 1]),
                                          torch.zeros_like(x[..., 1]))
            ground_impedance = modules.compl_div(
                modules.compl_mul(1 + torch.randn_like(y) * 1e-5, gt['ground_impedance']),
                (rho * c))
            diff_bc = dpdx2 + modules.compl_mul(modules.compl_div(y, ground_impedance),
                                                torch.tensor([0, 1]).to(y.device))

        sx = a0 * ((dist_west / Lpml) ** 2 + (dist_east / Lpml) ** 2)[..., None]
        sy = a0 * ((dist_north / Lpml) ** 2 + (dist_south / Lpml) ** 2)[..., None]

        ex = torch.cat((torch.ones_like(sx), -sx), dim=-1)
        ey = torch.cat((torch.ones_like(sy), -sy), dim=-1)

        A = modules.compl_div(ey, ex).repeat(1, 1, dpdx1.shape[-1] // 2)
        B = modules.compl_div(ex, ey).repeat(1, 1, dpdx1.shape[-1] // 2)
        C = modules.compl_mul(ex, ey).repeat(1, 1, dpdx1.shape[-1] // 2)

        pde1, _ = diff_operators.jacobian(modules.compl_mul(A, dpdx1), x)
        pde2, _ = diff_operators.jacobian(modules.compl_mul(B, dpdx2), x)

        pde1 = pde1[..., 0]
        pde2 = pde2[..., 1]
        # pde3 = modules.compl_mul(C, k ** 2 * y)
        pde3 = modules.compl_mul(C, (modules.compl_div(omega * torch.ones_like(c), c) ** 2).repeat(1, 1, y.shape[
            -1] // 2) * y)

        diff_constraint_hom = pde1 + pde2 + pde3

        diff_constraint_on = torch.where(torch.logical_and(source_boundary_values != 0., ~bc_mask),
                                         diff_constraint_hom - source_boundary_values,
                                         torch.zeros_like(diff_constraint_hom))
        diff_constraint_off = torch.where(torch.logical_and(source_boundary_values == 0., ~bc_mask),
                                          diff_constraint_hom,
                                          torch.zeros_like(diff_constraint_hom))

    def loss_bc(self, input):
        # x,y,t = input
        g = input.clone()
        g = self.scale_t(g)
        g.requires_grad = True
        r, phi = self.cylindrical_coords(input)
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)
        pnet = self.dnn(g)
        p_x_y_t = autograd.grad(pnet.view(-1, 1), g, torch.ones([input.view(-1, 3).shape[0], 1]).to(self.device),
                          create_graph=True)[0]
        p_x = p_x_y_t[:, [0]].flatten()
        p_y = p_x_y_t[:, [1]].flatten()
        dp_dt = p_x_y_t[:, [2]].flatten()
        dp_dr = sin_phi * p_y + cos_phi * p_x
        # Sommerfeld radiation condition (eq. 4.5.5 - "Acoustics" - Allan D. Pierce)
        # f = r * (dp_dr + 1 / self.c * dp_dt)
        f = r * (dp_dr + dp_dt)
        bcs_loss = self.loss_function(f.view(-1,1), torch.zeros_like(f.view(-1,1)))
        return bcs_loss

    def loss_ic(self, input):
        # x,y,t = input
        g = input.clone()
        g.requires_grad = True

        pnet = self.dnn(g)
        p_x_y_t = autograd.grad(pnet.view(-1, 1), g, torch.ones([input.view(-1, 3).shape[0], 1]).to(self.device),
                          create_graph=True)[0]
        dp_dt = p_x_y_t[:, [2]].flatten()
        f = pnet + dp_dt
        ics_loss = self.loss_function(f.view(-1,1), torch.zeros_like(f.view(-1,1)))
        return ics_loss

    def loss(self, input_data, input_pde, input_ic, pm):

        loss_p = self.loss_data(input_data, pm)
        loss_f = self.loss_PDE(input_pde)
        loss_bc = self.loss_bc(input_pde)
        loss_ic = self.loss_ic(input_ic)

        # loss_val = 1e2*loss_p + 1e-3*loss_f
        loss_val = self.lambda_data*loss_p + self.lambda_pde*loss_f + self.lambda_bc*loss_bc + 0.*loss_ic

        return loss_val, loss_p, loss_f, loss_bc, loss_ic

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

    def SGD_step(self, data_input, pde_input, ic_input, p_data):

        loss, loss_data, loss_pde, loss_bc, loss_ic = self.loss(data_input, pde_input, ic_input, p_data)

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
                loss_bc.cpu().detach().numpy(),
                loss_ic.cpu().detach().numpy())

    'test neural network'

    def test(self, test_input, p_true):
        g = test_input.clone().unsqueeze(0)
        g = self.scale_t(g)

        p_pred = self.dnn(g)
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

    def scale_t(self, input):
        x, y, t = input[:, 0, :].flatten(), input[:, 1, :].flatten(), input[:, 2, :].flatten()
        t = t*self.c
        return torch.stack((x, y, t), dim=-1).view(3, -1).unsqueeze(0)


class PINNDataset(Dataset):
    def __init__(self,
                 refdata,
                 measured_data,
                 x_ref,
                 y_ref,
                 x_m,
                 y_m,
                 t,
                 t_ind,
                 n_pde_samples = 800,
                 counter = 1):
        self.tfnp = lambda x : torch.from_numpy(x).float()
        self.counter = counter
        self.maxcounter = int(4e5)
        # self.maxcounter = -1
        self.TrainData = measured_data
        self.n_pde_samples = n_pde_samples
        # self.BCData = refdata[x_y_boundary_ind]
        self.t_ind = t_ind
        # self.x_y_boundary_ind = x_y_boundary_ind
        self.x_m = x_m
        self.y_m = y_m
        self.x_ref = x_ref
        self.y_ref = y_ref
        self.t = t
        self.tt = np.repeat(self.t, len(self.x_ref))
        self.xx = np.tile(self.x_ref, len(self.t))
        self.yy = np.tile(self.y_ref, len(self.t))
        self.collocation_all = self.tfnp(np.stack([self.xx, self.yy, self.tt], axis = 0))
        self.pressure_all = self.tfnp(refdata[:, self.t_ind].flatten())
        self.xmax = self.x_ref.max()
        self.xmin = self.x_ref.min()
        self.ymax = self.y_ref.max()
        self.ymin = self.y_ref.min()
        self.tmax = self.t[self.t_ind].max()
        self.counter_fun = lambda x, n : int(n * x)
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
            window_size = 100
            overlap = 50
            t_ind_windowed = window(self.t_ind,w = window_size, o = overlap) # 100 taps, 50 overlap
            n_windows = t_ind_windowed.shape[0]

            window_number = self.counter_fun(self.counter / self.maxcounter, n_windows)
            # t_ind_temp = self.t_ind[:(progressive_t_counter + 1)]
            t_ind_temp = t_ind_windowed[window_number]
            # idx = np.random.randint(0, progressive_t_counter + 1)
            idx = np.random.randint(0, window_size)
            t_batch_indx = t_ind_temp[idx]
        else:
            idx = np.random.randint(0, len(self.t_ind))
            t_batch_indx = self.t_ind[idx]
        t_data = self.t[t_batch_indx]
        pressure_batch = self.TrainData[:, t_batch_indx].flatten()
        pressure_bc_batch = self.TrainData[:, t_batch_indx].flatten()
        x_data, y_data = self.x_m, self.y_m
        x_pde = torch.FloatTensor(self.n_pde_samples).uniform_(self.xmin, self.xmax)
        y_pde = torch.FloatTensor(self.n_pde_samples).uniform_(self.ymin, self.ymax)
        x_bc = torch.FloatTensor(self.n_pde_samples).uniform_(self.xmin, self.xmax)
        y_bc = torch.FloatTensor(self.n_pde_samples).uniform_(self.ymin, self.ymax)
        x_ic = torch.FloatTensor(self.n_pde_samples).uniform_(self.xmin, self.xmax)
        y_ic = torch.FloatTensor(self.n_pde_samples).uniform_(self.ymin, self.ymax)
        t_ic = torch.zeros(self.n_pde_samples)

        if self.counter < self.maxcounter:
            t_pde = torch.FloatTensor(self.n_pde_samples).uniform_(0., t_data.max())
            t_bc = torch.FloatTensor(self.n_pde_samples).uniform_(0., t_data.max())
        else:
            t_pde = torch.FloatTensor(self.n_pde_samples).uniform_(0., self.tmax)
            t_bc = torch.FloatTensor(self.n_pde_samples).uniform_(0., t_data.max())
        tt_data = np.repeat(t_data, len(x_data))
        collocation_train = np.stack([x_data, y_data, tt_data], axis = 0)
        collocation_pde = np.stack([x_pde, y_pde, t_pde], axis = 0)
        collocation_bc = np.stack([x_bc, y_bc, t_bc], axis = 0)
        collocation_ic = np.stack([x_ic, y_ic, t_ic], axis = 0)
        self.counter += 1

        return {'collocation_train' : self.tfnp(collocation_train),
                'collocation_bc' : self.tfnp(collocation_bc),
                'collocation_pde' : self.tfnp(collocation_pde),
                'collocation_ic' : self.tfnp(collocation_ic),
                'pressure_bc_batch' : self.tfnp(pressure_bc_batch),
                'pressure_batch' : self.tfnp(pressure_batch),
                't_batch_indx' : t_batch_indx,
                'max_t': t_data.max()}

def window(a, w = 4, o = 2, copy = False):
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides = st, shape = sh)[0::o]
    if copy:
        return view.copy()
    else:
        return view
