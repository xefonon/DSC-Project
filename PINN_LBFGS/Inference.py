import torch
import numpy as np
from aux_functions import scan_checkpoint, load_checkpoint, FCN
import matplotlib.pyplot as plt
from pathlib import Path

device = 'cpu'

config = {
    'rir_time': 0.08,
    'lr': 1e-5,
    'layers': np.array([3, 100, 100, 100, 100, 100, 100, 100, 100, 100, 1]),  # 9 hidden layers
    'batch_size': 90,
    'siren': True,
}
bounds = {'x': (-1, 1),
          'y': (-1, 1),
          't': (0, config['rir_time'])}

PINN = FCN(config['layers'], bounds=bounds, device=device, siren=config['siren'])

checkpoint_dir = '/home/xen/PhD/Repositories/DSC-Project/PINNs/SIREN_torchmeta2'
cp_pinn = scan_checkpoint(checkpoint_dir, "PINN_")
state_dict_pinn = load_checkpoint(cp_pinn, device)
PINN.dnn.load_state_dict(state_dict_pinn["net"])
tfnp = lambda x: torch.from_numpy(x).float().to(PINN.device)

t = 343*np.linspace(0, config['rir_time'], 2000)
# y = np.linspace(-1, 1, 2000)
# t = 343*.05**np.ones_like(y)
# x = 0*np.ones_like(t)
x = 0.3*np.ones_like(t)
y = 0.2*np.ones_like(x)

input_coords = np.concatenate((x[..., None, None],y[..., None, None],t[..., None, None]), axis = 1)
out = PINN.dnn(tfnp(input_coords))

plt.plot(t/343, out.detach().cpu().numpy())
plt.show()
# %%
p = Path('../Data' + '/ISM_sphere.npz')
if not p.exists():
    p = Path('./Data/ISM_sphere.npz')
data = np.load(p.resolve())
keys = [key for key in data.keys()]
print("datafile keys: ", keys)
fs = data['fs']
rirdata = data['reference_data'][:, :int(fs*config['rir_time'])]
t = 343*np.linspace(0, config['rir_time'], len(rirdata.T))
grid = data['grid_reference']
idx = 50
x = grid[0, idx]*np.ones_like(t)
y = grid[1, idx]*np.ones_like(t)
input_coords = np.concatenate((x[..., None, None],y[..., None, None],t[..., None, None]), axis = 1)
out = PINN.dnn(tfnp(input_coords))

plt.plot(t/343, out.detach().cpu().numpy())
# plt.plot(t/343, rirdata[idx, :], alpha = 0.3)
plt.show()
