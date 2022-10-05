import sys
sys.path.append('../')
import numpy as np
from utils_soundfields import plot_sf
import re
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
import h5py

filename = '/home/xen/PhD/Repositories/DSC-Project/Data/SoundFieldControlPlanarDataset.h5'

def load_measurement_data(filename):

    with h5py.File(filename, "r") as f:
        data_keys = f.keys()
        meta_data_keys = f.attrs.keys()
        data_dict = {}
        for key in data_keys:
            data_dict[key] = f[key]
        for key in meta_data_keys:
            data_dict[key] = f.attrs[key]
        f.close()
    return data_dict