import sys
sys.path.append('../')
import torch
import torch.autograd as autograd  # computation graph
from torch.autograd import grad

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
from modules_meta import SingleBVPNet
from scipy import signal

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

def construct_input_vec(analytic, x_true, y_true, t, t_ind = None):
    if t_ind is not None:
        analytic = analytic[t_ind]
        t = t[t_ind]
    collocation = []
    for i in range(len(t)):
        tt = np.repeat(t[i], len(x_true))
        collocation.append(np.stack([x_true, y_true, tt], axis=0))
    return np.array(collocation), analytic

def plot_results(collocation_data, analytical, PINN):
    Nplots = collocation_data.shape[0]
    Pred_pressure = []
    tfnp = lambda x: torch.from_numpy(x).float().to(PINN.device)
    for n in range(Nplots):
        p_pred = PINN.test(tfnp(collocation_data[n]))
        Pred_pressure.append(p_pred.squeeze(0))
    fig, axes = plt.subplots(nrows=2, ncols=Nplots, sharex=True, sharey=True)
    p_pred_minmax = (np.array(Pred_pressure).min(), np.array(Pred_pressure).min()+ 1.)
    p_true_minmax = (analytical.min(), analytical.mean())
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
        ax, im2 = plot_sf(analytical[i], collocation_data[i, 0], collocation_data[i, 1],
                        ax=ax, name = name,
                        clim = p_true_minmax)
        if i != 0:
            ax.set_ylabel('')
    fig.subplots_adjust(right=0.8)
    # pressure colorbar
    cbar_ax = fig.add_axes([0.82, 0.61, 0.02, 0.2])
    fig.colorbar(im, cax=cbar_ax)
    cbar_ax2 = fig.add_axes([0.82, 0.15, 0.02, 0.2])
    fig.colorbar(im2, cax=cbar_ax2)

    return fig

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
            self.net = SingleBVPNet(out_features= 1, in_features= 3, hidden_features= 256, num_hidden_layers= 5)
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
        x, y, t = g[:, :,0].flatten(), g[:,:, 1].flatten(), g[:,:, 2].flatten()

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

    def wave_pml(self, input_data, model_output, gt):
        source_boundary_values = gt['source_boundary_values'].to(self.device)
        x = input_data  # (meta_batch_size, num_points, 3)
        y = model_output  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask'].to(self.device)
        batch_size = x.shape[1]

        du, status = jacobian(y, x)
        dudt = du[..., 2]

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            hess, status = jacobian(du[..., 0, :], x)
            lap = hess[..., 0, 0, None] + hess[..., 1, 1, None]
            dudt2 = hess[..., 2, 2, None]
            diff_constraint_hom = dudt2 - 1. * lap

        dirichlet = y.unsqueeze(-1)[dirichlet_mask] - source_boundary_values[dirichlet_mask]
        neumann = dudt[:,0][dirichlet_mask.squeeze(-1)]

        return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 1e2,
                'neumann': torch.abs(neumann).sum() * batch_size / 1e2,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    def loss(self, input_data, gt):
        xyt = input_data['coords'].clone().to(self.device)
        xyt = self.scale_t(xyt)
        xyt.requires_grad = True

        loss_dict = self.wave_pml(xyt, self.dnn(xyt), gt)
        loss_dir, loss_neu, loss_wave = loss_dict['dirichlet'], loss_dict['neumann'], loss_dict['diff_constraint_hom']
        loss_val = loss_dir + loss_neu + loss_wave
        # loss_val = 1e2*loss_p + 1e-3*loss_f
        # loss_val = self.lambda_data*loss_p + self.lambda_pde*loss_f + self.lambda_bc*loss_bc

        return loss_val, loss_dir, loss_neu, loss_wave

    'callable for optimizer'


    def SGD_step(self, input_data, gt):

        loss, loss_dir, loss_neu, loss_wave = self.loss(input_data, gt)

        loss.backward()


        return (loss.cpu().detach().numpy(),
                loss_dir.cpu().detach().numpy(),
                loss_neu.cpu().detach().numpy(),
                loss_wave.cpu().detach().numpy())

    'test neural network'

    def test(self, xyt):
        xyt = self.scale_t(xyt.T.unsqueeze(0))

        p_pred = self.dnn(xyt)
        # Relative L2 Norm of the error
        # relative_error = torch.linalg.norm((p_true - p_pred.squeeze(0)), 2) / torch.linalg.norm(p_true,2)
        # Error vector
        p_pred = p_pred.cpu().detach().numpy()

        return p_pred

    def scale_xy(self, input):
        x, y, t = input[:, 0, :].flatten(), input[:, 1, :].flatten(), input[:, 2, :].flatten()
        x = x/self.c
        y = y/self.c
        return torch.stack((x, y, t), dim=-1).view(3, -1).unsqueeze(0)

    def scale_t(self, input):
        x, y, t = input[:, :, 0].flatten(), input[:, :,  1].flatten(), input[:, :,  2].flatten()
        t = t*self.c
        return torch.stack((x, y, t), dim=-1).view(-1, 3).unsqueeze(0)


class PINNDataset(Dataset):
    def __init__(self,
                 rirdata,
                 x_true,
                 y_true,
                 t,
                 data_ind,
                 x_y_boundary_ind,
                 t_ind,
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

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords

def gaussian(x, mu=[0, 0], sigma=1e-4, d=2):
    x = x.numpy()
    if isinstance(mu, torch.Tensor):
        mu = mu.numpy()

    q = -0.5 * ((x - mu) ** 2).sum(1)
    return torch.from_numpy(1 / np.sqrt(sigma ** d * (2 * np.pi) ** d) * np.exp(q / sigma)).float()


def hessian(y, x):
    ''' hessian of y wrt x
    y: shape (meta_batch_size, num_observations, channels)
    x: shape (meta_batch_size, num_observations, 2)
    '''
    meta_batch_size, num_observations = y.shape[:2]
    grad_y = torch.ones_like(y[..., 0]).to(y.device)
    h = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1], x.shape[-1]).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        dydx = grad(y[..., i], x, grad_y, create_graph=True)[0]

        # calculate hessian on y for each x value
        for j in range(x.shape[-1]):
            h[..., i, j, :] = grad(dydx[..., j], x, grad_y, create_graph=True)[0][..., :]

    status = 0
    if torch.any(torch.isnan(h)):
        status = -1
    return h, status

def jacobian(y, x):
    ''' jacobian of y wrt x '''
    meta_batch_size, num_observations = y.shape[:2]
    jac = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1]).to(y.device) # (meta_batch_size*num_points, 2, 2)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        y_flat = y[...,i].view(-1, 1)
        jac[:, :, i, :] = grad(y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1

    return jac, status

class WaveSource(Dataset):
    def __init__(self, sidelength, source_coords=[0., 0., 0.], pretrain=False):
        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.sidelength = sidelength
        self.mgrid = get_mgrid(self.sidelength).detach()

        self.N_src_samples = 100
        self.sigma = 5e-4
        self.source_coords = torch.tensor(source_coords).view(-1, 3)

        self.counter = 0
        self.full_count = 100e3

    def __len__(self):
        return 1


    def __getitem__(self, idx):
        start_time = self.source_coords[0, 0]  # time to apply  initial conditions

        r = 5e2 * self.sigma * torch.rand(self.N_src_samples, 1).sqrt()
        phi = 2 * np.pi * torch.rand(self.N_src_samples, 1)

        # circular sampling
        source_coords_x = r * torch.cos(phi) + self.source_coords[0, 1]
        source_coords_y = r * torch.sin(phi) + self.source_coords[0, 2]
        source_coords = torch.cat((source_coords_x, source_coords_y), dim=1)

        # uniformly sample domain and include coordinates where source is non-zero
        coords = torch.zeros(self.sidelength ** 2, 2).uniform_(-1, 1)

        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.zeros(self.sidelength ** 2, 1).uniform_(start_time - 0.001, start_time + 0.001)
            coords = torch.cat((coords, time), dim=1)
            # make sure we spatially sample the source
            coords[-self.N_src_samples:, :2] = source_coords
        else:
            # slowly grow time values from start time
            # this currently assumes start_time = 0 and max time value is 0.75.
            time = torch.zeros(self.sidelength ** 2, 1).uniform_(0, 0.4 * (self.counter / self.full_count))
            coords = torch.cat((coords, time), dim=1)

            # make sure we always have training samples at the initial condition
            coords[-self.N_src_samples:, :2] = source_coords
            coords[-2 * self.N_src_samples:, 2] = start_time

            # set up source
        normalize = 50 * gaussian(torch.zeros(1, 2), mu=torch.zeros(1, 2), sigma=self.sigma, d=2)
        boundary_values = gaussian(coords[:, :2], mu=self.source_coords[:, :2], sigma=self.sigma, d=2)[:, None]
        boundary_values /= normalize

        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            boundary_values = torch.where((coords[:, 2, None] == start_time), boundary_values, torch.Tensor([0]))
            dirichlet_mask = (coords[:, 2, None] == start_time)

        boundary_values[boundary_values < 1e-5] = 0.

        self.counter += 1

        if self.pretrain and self.counter == 2000:
            self.pretrain = False
            self.counter = 0

        return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}


class GreensFunAnalytical:
    def __init__(self, xlim = 1, ylim = 1, fs = 8e3, npoints = 100,
                 spatial_sigma = 1e-3,
                 temporal_sigma = 50,
                 onset_time = 0.0001):
        self.spatial_sigma  = spatial_sigma   # spatial width of the Gaussian
        self.temporal_sigma = temporal_sigma  # temporal width of the Gaussian
        self.onset_time     = onset_time # time when Gaussian Disturbance reaches its peak

        self.x = np.linspace(-xlim, xlim, npoints)
        self.y = np.linspace(-ylim, ylim, npoints)
        self.t = np.linspace(0, 1, fs)

    # Equation to solve
    # phi(x,y,t) = ∫∫∫ G(x,y,t; x',y',t') . Source(x',y',t') dx' dy' dt'
    # G(x,y,t; x',y',t') = Green's Function
    # phi = displacement by the wave

    # Define Function to realize Green's Function for Wave Equation
    def gw(self, xx, yy, tt):

        kk = np.heaviside((tt-np.sqrt(xx**2+yy**2)),1)/(2*np.pi*np.sqrt(np.clip(tt**2-xx**2-yy**2,0,None))+1)
        return (kk)

    def gaussian(self, x, mu=[0, 0], sigma=1e-4, d=2):
        q = -0.5 * ((x - mu) ** 2).sum(1)
        return 1 / np.sqrt(sigma ** d * (2 * np.pi) ** d) * np.exp(q / sigma)

    # Define Function to realize Gaussian Disturbance
    def g_source(self, xx, yy, tt):
        # coords = np.concatenate((xx[0],yy[:, :, 0]), axis = 0).T
        # normalize = 50 * self.gaussian(np.zeros((1, 2)), mu=np.zeros((1, 2)), sigma=self.sigma, d=2)
        # boundary_values = self.gaussian(coords[:, :2], mu=[0.,0.], sigma=self.sigma, d=2)[:, None]
        # boundary_values /= normalize

        kk = np.exp((-(np.sqrt((xx)**2+(yy)**2)/self.spatial_sigma)**2))*\
             np.exp(-((tt-self.onset_time)/self.temporal_sigma)**2)
        return (kk)
    def calculate(self):
        # Calculate the two function for given grid points
        green = self.gw(self.x[None,None,:],self.y[None,:,None],self.t[:,None,None])
        gauss = self.g_source(self.x[None,None,:],self.y[None,:,None],self.t[:,None,None])

        # Calculate Source Response via convolution
        phi = signal.convolve(gauss, green, mode='same')
        return phi # tsteps, xsteps, ysteps
    def return_points(self):
        X, Y = np.meshgrid(self.x, self.y)
        return np.concatenate((X.flatten()[..., None], Y.flatten()[..., None]), axis = -1).T, self.t