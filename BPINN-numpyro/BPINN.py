import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from torch import nn
import torch
import numpy as np
from pyro.infer import SVI, Trace_ELBO, MCMC, NUTS, HMC
from pyro.contrib.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
from pyro.infer import Predictive


def gradients(self, outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]


class BPINN(PyroModule):
    def __init__(self, h0=3, h1=50, h2=50, h3=50, h4=50, sigma=1, device='cuda'):
        super().__init__()
        # sigma = torch.tensor(sigma, device = device)
        # sigma = PyroSample(dist.InverseGamma(3.,.1))
        # sigma = dist.InverseGamma(3.,.1).sample()
        self.fc1 = PyroModule[nn.Linear](h0, h1)
        self.fc1.weight = PyroSample(dist.Normal(0., sigma).expand([h1, h0]).to_event(2))
        self.fc1.weight = self.fc1.weight.to(device)
        self.fc1.bias = PyroSample(dist.Normal(0., sigma).expand([h1]).to_event(1))
        self.fc1.bias = self.fc1.bias.to(device)
        self.fc2 = PyroModule[nn.Linear](h1, h2).to(device)
        self.fc2.weight = PyroSample(dist.Normal(0., sigma).expand([h2, h1]).to_event(2))
        self.fc2.weight = self.fc2.weight.to(device)
        self.fc2.bias = PyroSample(dist.Normal(0., sigma).expand([h2]).to_event(1))
        self.fc2.bias = self.fc2.bias.to(device)
        self.fc3 = PyroModule[nn.Linear](h2, h3).to(device)
        self.fc3.weight = PyroSample(dist.Normal(0., sigma).expand([h3, h2]).to_event(2))
        self.fc3.weight = self.fc3.weight.to(device)
        self.fc3.bias = PyroSample(dist.Normal(0., sigma).expand([h3]).to_event(1))
        self.fc3.bias = self.fc3.bias.to(device)
        self.fc4 = PyroModule[nn.Linear](h3, h4).to(device)
        self.fc4.weight = PyroSample(dist.Normal(0., sigma).expand([h4, h3]).to_event(2))
        self.fc4.weight = self.fc4.weight.to(device)
        self.fc4.bias = PyroSample(dist.Normal(0., sigma).expand([h4]).to_event(1))
        self.fc4.bias = self.fc4.bias.to(device)
        self.fc5 = PyroModule[nn.Linear](h4, 1).to(device)
        self.fc5.weight = PyroSample(dist.Normal(0., sigma).expand([1, h4]).to_event(2))
        self.fc5.weight = self.fc5.weight.to(device)
        self.fc5.bias = PyroSample(dist.Normal(0., sigma).expand([1]).to_event(1))
        self.fc5.bias = self.fc5.bias.to(device)
        self.act = nn.Tanh()

    def forward(self, x, y=None):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        mu = self.fc5(x).squeeze()
        return mu


class Model_BPINN(PyroModule):
    def __init__(self, data_params, nn_params):
        super().__init__()
        self.net = BPINN(**nn_params)  # this is a PyroModule
        # self.net_prec = PyroSample(dist.Gamma(100.0, 1.0))
        # self.net_scale = 1.0 / torch.sqrt(self.net_prec)
        # self.f_prec = PyroSample(dist.Gamma(100.0, 1.0))
        # self.f_scale = 1.0 / torch.sqrt(self.f_prec)
        self.net_scale = 0.001
        self.f_scale = 0.001
        self.device = nn_params['device']
        self.c = data_params['speed_of_sound']
        self.batch_size = data_params['batch_size']

        self.x_min, self.x_max = data_params['x_bounds']
        self.y_min, self.y_max = data_params['y_bounds']
        self.t_min, self.t_max = data_params['t_bounds']

        self.ub = torch.Tensor([self.x_max, self.y_max, self.t_max]).to(nn_params['device'])
        self.lb = torch.Tensor([self.x_min, self.y_min, self.t_min]).to(nn_params['device'])

        self.scaling = lambda x: 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        self.batch_counter = 0

    def scale_t(self, input):
        x = input.select(1, 0).view(-1, 1)
        y = input.select(1, 1).view(-1, 1)
        t = input.select(1, 2).view(-1, 1)
        t = t * self.c
        return torch.cat([x, y, t], dim=1)

    def network_scaling(self, input):
        x = input.select(1, 0).view(-1, 1)
        y = input.select(1, 1).view(-1, 1)
        t = input.select(1, 2).view(-1, 1)
        t = t * self.c
        return torch.cat([x, y, t], dim=1)

    def forward(self, Xdata, Xpde=None, p_measured=None, f=None):

        # data likelihood
        Xdata = self.scale_t(Xdata)
        Xdata = self.scaling(Xdata)

        xdata = Xdata.select(1, 0).view(-1, 1)
        ydata = Xdata.select(1, 1).view(-1, 1)
        tdata = Xdata.select(1, 2).view(-1, 1)
        tdata.requires_grad_(True)
        xdata.requires_grad_(True)
        ydata.requires_grad_(True)

        p_mu = self.net(torch.cat([xdata, ydata, tdata], dim=1))
        sigma = pyro.sample("sigma_obs", dist.InverseGamma(3., 1.))
        sigma = sigma.to(self.device)
        # prec_obs = pyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
        # sigma = 1.0 / torch.sqrt(prec_obs).to(self.device)

        with pyro.plate("observations", Xdata.shape[0], subsample_size=self.batch_size) as ind:
            p_hat = pyro.sample("obs1", dist.Normal(p_mu[ind], sigma).to_event(1), obs=p_measured[ind])

        # PDE likelihood
        if Xpde is not None:
            Xpde = self.scale_t(Xpde)
            Xpde = self.scaling(Xpde)
            xpde = Xpde.select(1, 0).view(-1, 1)
            ypde = Xpde.select(1, 1).view(-1, 1)
            tpde = Xpde.select(1, 2).view(-1, 1)
            tpde.requires_grad_(True)
            xpde.requires_grad_(True)
            ypde.requires_grad_(True)

            p_mu_pde = self.net(torch.cat([xpde, ypde, tpde], dim=1))
            p_t = self.gradient(p_mu_pde, tpde)
            p_tt = self.gradient(p_t, tpde)
            p_x = self.gradient(p_mu_pde, xpde)
            p_xx = self.gradient(p_x, xpde)
            p_y = self.gradient(p_mu_pde, ypde)
            p_yy = self.gradient(p_y, ypde)

            f_sigma = pyro.sample("f_sigma",dist.InverseGamma(3., 1.))
            f_sigma = f_sigma.to(self.device)
            # prec_pde = pyro.sample("prec_pde", dist.Gamma(3.0, 1.0))
            # f_sigma = 1.0 / torch.sqrt(prec_pde).to(self.device)
            with pyro.plate("PDE", Xpde.shape[0]):
                # f_mu = p_xx + p_yy - (1./self.c**2)*p_tt
                f_mu = p_xx + p_yy - 1. * p_tt
                f_hat = pyro.sample("obs2", dist.Normal(f_mu, f_sigma).to_event(1), obs=f)

            return p_hat, f_hat
        else:
            return p_hat

    def gradient(self, outputs, inputs):
        outputs_sum = outputs.sum()
        outputs_sum.requires_grad_(True)
        return torch.autograd.grad(outputs_sum, inputs, retain_graph=True, create_graph=True)[0]

    def test(self, inputdata, outputdata, n_samples = 1000):
        inputdata = self.scale_t(inputdata)
        inputdata = self.scaling(inputdata)

        xdata = inputdata.select(1, 0).view(-1, 1)
        ydata = inputdata.select(1, 1).view(-1, 1)
        tdata = inputdata.select(1, 2).view(-1, 1)
        p_mu = self.net(torch.cat([xdata, ydata, tdata], dim=1))
        sigma = pyro.sample("sigma_test", dist.InverseGamma(3., 1.))
        sigma = sigma.to(self.device)

        with pyro.plate("observations", n_samples):
            p_hat = pyro.sample("test_samples", dist.Normal(p_mu, sigma).to_event(1))
        p_hat_mu = p_hat.mean(axis = 0)
        relative_error = (torch.abs(outputdata - p_hat_mu)**2).mean()
        # Error vector
        error_vec = torch.abs(outputdata - p_hat_mu / (outputdata + np.finfo(np.float32).eps))
        p_pred = p_hat_mu.cpu().detach().numpy()
        error_vec = error_vec.cpu().detach().numpy()

        return error_vec, relative_error.item(), p_pred
