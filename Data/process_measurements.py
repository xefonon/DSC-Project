import sys
sys.path.append('../')
import numpy as np
from glob import glob
from utils_soundfields import plot_sf
import re
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
import h5py

# %%
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]
# %%

folder_path = '/home/xen/PhD/Repositories/DSC-Project/Data/measurement_data_odeon_src'

glob_bn_noise = glob(folder_path+'/background_*.npz')
glob_measurements = glob(folder_path+'/soundfieldcontrol_rirs_rir_*.npy')
glob_metadata = glob(folder_path+'/soundfieldcontrol_rirs_meta_*.npy')

meta_data_dict = np.load(glob_metadata[0], allow_pickle=True).item()


measurements = np.load(glob_metadata[0], allow_pickle=True).item()
noise_recs = []
for file in glob_bn_noise:
    noise_item = np.load(file, allow_pickle=True)
    noise_recs.append(noise_item['noise'])
grid = meta_data_dict['measurement_grid']
fs = meta_data_dict['fs']
glob_measurements.sort(key=natural_keys)

measurements = []
for filename in glob_measurements:
    tempfile = np.load(filename)
    measurements.append(tempfile)

measurements = np.array(measurements)
# truncate
measurements = measurements[:, 180800:240000]

# %%

# folder_path2 = '/home/xen/PhD/Repositories/DSC-Project/Data/measurement_data_top_plane'
#
# glob_bn_noise2 = glob(folder_path2+'/background_*.npz')
# glob_measurements2 = glob(folder_path2+'/soundfieldcontrol_rirs_rir_*.npy')
# glob_metadata2 = glob(folder_path2+'/soundfieldcontrol_rirs_meta_*.npy')
#
# meta_data_dict2 = np.load(glob_metadata2[0], allow_pickle=True).item()
#
#
# measurements2 = np.load(glob_metadata2[0], allow_pickle=True).item()
# noise_recs2 = []
# for file in glob_bn_noise2:
#     noise_item2 = np.load(file, allow_pickle=True)
#     noise_recs2.append(noise_item2['noise'])
# grid2 = meta_data_dict2['measurement_grid']
# fs = meta_data_dict['fs']
# glob_measurements2.sort(key=natural_keys)
#
# measurements2 = []
# for filename in glob_measurements2:
#     tempfile = np.load(filename)
#     measurements2.append(tempfile)
#
# measurements2 = np.array(measurements2)
# # truncate
# measurements2 = measurements2[:, 180800:240000]

#%%
t = np.linspace(0, measurements.shape[-1]/fs, measurements.shape[-1])
# translate axes
ref_t = meta_data_dict['reference_position_room_coord']
ref = meta_data_dict['reference_position']
new_coord_origin = ref_t + ref
r0 = ref_t - ref
grid = grid + r0[..., None]
# grid2 = grid2 + r0[..., None]
# %%
t = t[int(0.25*fs):int(0.58*fs)]
measurements = measurements[:, int(0.25*fs):int(0.58*fs)]
# measurements2 = measurements2[:, int(0.25*fs):int(0.58*fs)]
#%%
hf = h5py.File('SoundFieldControlPlanarDataset_src2.h5', 'w')
# hf.create_dataset('Noise_recs_top',  compression="gzip", data=noise_recs2)
hf.create_dataset('Noise_recs_bottom',  compression="gzip", data=noise_recs)
# dset = hf.create_dataset('RIRs_top',  compression="gzip", data=measurements2)
dset1 = hf.create_dataset('RIRs_bottom',  compression="gzip", data=measurements)
dset2 = hf.create_dataset('sweep_signal',  compression="gzip", data=meta_data_dict['sweep_signal'])
# hf.attrs['grid_top'] = grid2
hf.attrs['fs'] = meta_data_dict['fs']
hf.attrs['grid_bottom'] = grid
hf.attrs['sweep_duration'] = meta_data_dict['sweep_duration']
hf.attrs['sweep_range'] = meta_data_dict['sweep_range']
hf.attrs['sweep_amplitude'] = meta_data_dict['sweep_amplitude']
hf.attrs['calibration'] = meta_data_dict['calibration']
hf.attrs['true_calibration'] = meta_data_dict['true_calibration']
hf.attrs['temperature'] = meta_data_dict['temperature']
hf.attrs['loudspeaker_position'] = meta_data_dict['loudspeaker_position']
hf.attrs['room_dimensions'] = meta_data_dict['room_dimensions']

hf.close()
# %%

plt.plot(t, measurements[0])
plt.xlabel('time [s]')
plt.show()
# %%
t_plot = .006
ind = int(fs*t_plot)
plot_sf(measurements[:,ind], grid[0], grid[1], colorbar= True, interpolated= True)
plt.show()
plot_sf(measurements[:,ind], grid[0], grid[1], colorbar= True, interpolated= False)
plt.show()
# %%
t_plot = .012
ind = int(fs*t_plot)

plot_sf(measurements[:,ind], grid[0], grid[1], colorbar= True, interpolated= False)
plt.show()


#%% ANIMATE

anim_meas = measurements[:, int(0.27*fs):int(0.30*fs):int(0.0005*fs)]

t_anim = np.linspace(0, measurements[:, int(0.27*fs):int(0.30*fs)].shape[-1]/fs,
                     measurements[:, int(0.27*fs):int(0.30*fs)].shape[-1])
t_anim = t_anim[0:-1:int(0.0005*fs)]

fig = plt.figure()
ax1 = fig.add_subplot()
clims = (anim_meas.min(),anim_meas.max())
# animation function
def animate(i):
    # t is a parameter
    # tt = t_anim[i]

    # x, y values to be plotted
    plot_sf(anim_meas[:,i], grid[0], grid[1], colorbar=False, normalise= False, interpolated=True,
            ax = ax1, clim = clims)
    ax1.set_title(f't : {t_anim[i]:.3f} s')
    # appending new points to x, y axes points list
ani = FuncAnimation(fig, animate, interval=1, frames=anim_meas.shape[-1],)
# Save the animation as an animated GIF
ani.save("sound_field.gif", dpi=100,
         writer=PillowWriter(fps=5))
# ani.save("soundfield.gif",  dpi=100,  writer = "imagemagick")