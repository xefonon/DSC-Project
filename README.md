# DSC Project

This repo is for the Danish Sound Cluster project named 
"Physics-Informed neural networks for sound field reconstruction", a collaboration between
DTU, GN Audio, Odeon and HBK.

The repo is still at an early stage so please be patient, or feel free to clone or create a new branch.


Make sure you have Anaconda installed to create a new environment from the command

`` conda env create --name envname --file=environment.yml
``

Simulate data by modifying the script (if required) and running

``python Data/ISM_spherical_array.py``

Train the neural network by running

``python PINNs/run_PINN.py``

Current Repo Tasks
: The following tasks will be carried out here.

- [x] Create repository
- [x] Include PINN 
- [x] Simulate Data
- [ ] Examine convergence
- [ ] Use with experimental data
