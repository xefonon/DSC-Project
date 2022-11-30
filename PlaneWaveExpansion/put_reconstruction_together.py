import numpy as np
from glob import glob
import re
import sys
sys.path.append('./PlaneWaveExpansion')
import torch
sys.path.append('./PINNs')
from aux_functions import scan_checkpoint, load_checkpoint, FCN, get_measurement_vectors, maxabs_normalize_rirs
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks
import scienceplots
plt.style.use(['science','ieee', 'std-colors'])

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]
def correct_rir_alignment(rir, lag, forward = True):
    if forward:
        return np.pad(rir, (lag, 0), 'constant')[:-lag]
    else:
        return np.pad(rir[lag:], (0, lag), 'constant')


def sinc_interp(x, s, u):
    """
    Interpolates x, sampled at "s" instants
    Output y is sampled at "u" instants ("u" for "upsampled")

    from Matlab:
    http://phaseportrait.blogspot.com/2008/06/sinc-interpolation-in-matlab.html
    """

    if len(x) != len(s):
        raise Exception('x and s must be the same length')

    # Find the sampling period of the undersampled signal
    T = s[1] - s[0]

    y = []
    for i in range(len(u)):
        y.append(np.sum(np.multiply(x, np.sinc((1 / T) * (u[i] - s)))))

    y = np.reshape(y, len(u))
    return y

# %%
# import matplotlib.image as mpimg
# img = mpimg.imread('./PlaneWaveExpansion/cameraman.png')
#
# M = len(img)
# dt = 0.1
# # t = np.arange(1, M + 1)
# # ts = np.arange(-M, 2*M +dt, dt)
t = np.arange(-M/2, M/2)
ts = np.arange(-M/2, M/2, dt)

Ts,T = np.meshgrid(ts, t, indexing='ij')
A=np.sinc(Ts - T)

index1 = np.arange(0, M/dt)
index2 = np.arange(M + 1 - M/dt, M + 1)

tempA = A[A[:,0].argmax():]

imgnew = tempA@img@tempA.T
plt.imshow(imgnew)
plt.show()
# %%
# for time in range(pm.shape[-1]):
# %%

files = glob("PlaneWaveExpansion/Reconstructions/reconstructed_pressure_partion_n_*.npz")

files.sort(key=natural_keys)

TF = []

for f in files:
    TF.append(np.load(f, allow_pickle=True)['arr_0'])

TF = np.array(TF)
prec = np.fft.irfft(TF.T)
# %%
filename = './Data/SoundFieldControlPlanarDataset.h5'
pref, fs, grid, pm, grid_measured, c = get_measurement_vectors(filename=filename)
_, _, _, pref, grid, c = get_measurement_vectors(filename=filename, subsample_points = 30)
# %% sinc interpolation
ind = int(0.0145*8000)
pm_sinc = pm[:, ind].reshape(10, 10)
pref_sinc = pref[:, ind].reshape(30, 30)
M = pm_sinc.shape[0]
t = np.arange(-M/2, M/2)
ts = np.linspace(-M/2, M/2, 30)

Ts,T = np.meshgrid(ts, t, indexing='ij')
A=np.sinc(Ts - T)

tempA = A[A[:,0].argmax():]

imgnew = tempA@pm_sinc@tempA.T
plt.imshow(pref_sinc)
plt.show()
plt.imshow(pm_sinc)
plt.show()

plt.imshow(imgnew)
plt.show()


#%%
device = 'cpu'

l_inf_norm = 1
scaler = maxabs_normalize_rirs(pref, device=device, l_inf_norm=l_inf_norm)

config = {
    'rir_time': 0.1,
    'n_hidden_layers': 3,  # 3 hidden layers
    'siren': True,
}
bounds = {
    'x': (-1, 1),
    'y': (-1, 1),
    't': (0, config['rir_time'])}

# data = measureddata[:, int(0.003 * fs):int(config['rir_time'] * fs)]  # truncate
# refdata = pref[:, int(0.003 * fs):int(config['rir_time'] * fs)]  # truncate
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
cp_pinn = checkpoint_dir + '/wave_eq_100_sensors'
# cp_pinn = scan_checkpoint(checkpoint_dir, "PINN_")
state_dict_pinn = load_checkpoint(cp_pinn, device)
PINN.dnn.load_state_dict(state_dict_pinn["net"])
tfnp = lambda x: torch.from_numpy(x).float().to(PINN.device)

# %%
# plot index
plt_indx = 29
prec_rir = prec[plt_indx]
pref_rir = pref[plt_indx]

corr = np.correlate(prec_rir - np.mean(prec_rir),
                    pref_rir - np.mean(pref_rir),
                    mode='full')


t = np.linspace(0, prec.shape[-1]/fs, prec.shape[-1])

init = int(fs*0.01)
trunc = int(fs*0.06)
prec_rir = prec_rir[init:trunc]/np.max(abs(prec_rir[init:trunc]))
pref_rir = pref_rir[init:trunc]/np.max(abs(pref_rir[init:trunc]))

t_plane_wave =  np.linspace(0, prec_rir.shape[0]/fs, prec_rir.shape[0])
tnet = t[init:trunc]

x = grid[0, plt_indx] * np.ones_like(t)
y = grid[1, plt_indx] * np.ones_like(t)

input_coords = np.concatenate((x[..., None, None], y[..., None, None], c*t[..., None, None]), axis=1)
out = PINN.dnn(tfnp(input_coords))
net_init = int(fs*(0.01 - 0.003))
net_trunc = int(fs*(0.06 - 0.003)) + 1
netout = out.detach().cpu().numpy()
netout = netout[net_init:net_trunc]
# %%

prec_net = (netout/np.max(abs(netout))).squeeze(1)

pref_net_rir = pref[plt_indx, init:trunc]/np.max(abs(pref[plt_indx, init:trunc])).squeeze(-1)


fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.plot(t_plane_wave, pref_rir, label = 'True')
ax.plot(t_plane_wave, prec_rir, label = 'Plane Wave Expansion (Reconstructed)', alpha = 0.8)
ax.set_xlim(t_plane_wave.min(), t_plane_wave.max())
ax.legend()
ax = fig.add_subplot(2, 1, 2)
ax.plot(t_plane_wave, pref_net_rir, label = 'True')
ax.plot(t_plane_wave, prec_net, label = 'PINN (Reconstructed)', alpha = 0.8)
ax.legend()
ax.set_xlim(t_plane_wave.min(), t_plane_wave.max())
ax.autoscale(tight=True)
fig.show()
# fig.savefig('comparison_pinn_pwe.png', dpi = 300)

