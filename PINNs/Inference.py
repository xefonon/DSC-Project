import torch
import numpy as np
import sys
sys.path.append('./PINNs')
from aux_functions import scan_checkpoint, load_checkpoint, FCN, get_measurement_vectors, maxabs_normalize_rirs
import matplotlib.pyplot as plt
from pathlib import Path

#%%
device = 'cpu'
data_dir = './Data'
filename = data_dir + '/SoundFieldControlPlanarDataset.h5'
refdata, fs, grid, measureddata, grid_measured, c = get_measurement_vectors(filename,
                                                                            real_data=True,
                                                                            subsample_points=25)  # per dimension

l_inf_norm = 1
scaler = maxabs_normalize_rirs(refdata, device=device, l_inf_norm=l_inf_norm)

config = {
    'rir_time': 0.1,
    'n_hidden_layers': 3,  # 3 hidden layers
    'siren': True,
}
bounds = {
    'x': (-1, 1),
    'y': (-1, 1),
    't': (0, config['rir_time'])}

data = measureddata[:, int(0.003 * fs):int(config['rir_time'] * fs)]  # truncate
refdata = refdata[:, int(0.003 * fs):int(config['rir_time'] * fs)]  # truncate
#%%
PINN = FCN(n_hidden_layers=config['n_hidden_layers'],
           device=device,
           siren=config['siren'],
           loss_fn='mae',
           c=c,
           output_scaler=scaler,
           fs=fs,
           bounds= bounds,
           map_input= False
           )

#%%
checkpoint_dir = './PINNs/PINN_checkpoints'
cp_pinn = scan_checkpoint(checkpoint_dir, "PINN_")
state_dict_pinn = load_checkpoint(cp_pinn, device)
PINN.dnn.load_state_dict(state_dict_pinn["net"])
tfnp = lambda x: torch.from_numpy(x).float().to(PINN.device)
# %%
plot_index = -5
# t = c * np.linspace(0, config['rir_time'], 2000)
t_plot = 0.09
t = c * np.linspace(0, t_plot, int(t_plot * fs))
# y = np.linspace(-1, 1, 2000)
# t = 343*.05**np.ones_like(y)
# x = 0*np.ones_like(t)
x = grid[0, plot_index] * np.ones_like(t)
y = grid[1, plot_index] * np.ones_like(x)

input_coords = np.concatenate((x[..., None, None], y[..., None, None], t[..., None, None]), axis=1)
out = PINN.dnn(tfnp(input_coords))

netout = out.detach().cpu().numpy()
netout = netout/np.max(abs(netout))

reference = refdata[plot_index, :int(t_plot * fs)]
reference = reference/np.max(abs(reference))
plt.plot(t / c, netout)
plt.plot(t / c, reference, alpha = 0.7)
plt.show()
# %%
p = Path('./Data' + '/ISM_sphere.npz')
if not p.exists():
    p = Path('./Data/ISM_sphere.npz')
data = np.load(p.resolve())
keys = [key for key in data.keys()]
print("datafile keys: ", keys)
fs = data['fs']
rirdata = data['reference_data'][:, :int(fs * config['rir_time'])]
t = c * np.linspace(0, config['rir_time'], len(rirdata.T))
grid = data['grid_reference']
idx = 50
x = grid[0, idx] * np.ones_like(t)
y = grid[1, idx] * np.ones_like(t)
input_coords = np.concatenate((x[..., None, None], y[..., None, None], t[..., None, None]), axis=1)
out = PINN.dnn(tfnp(input_coords))

plt.plot(t / c, out.detach().cpu().numpy())
# plt.plot(t/343, rirdata[idx, :], alpha = 0.3)
plt.show()
