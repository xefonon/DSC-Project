#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J SIREN_helmholtz
### -- ask for number of cores (default: 1) --
#BSUB -n 2
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
# request 24GB of system-memory
#BSUB -R "rusage[mem=24GB]"
### -- send notification at start -- 
### -- send notification at completion -- 
#BSUB -N 
#BSUB -o ./log/log-%J-%I.out
#BSUB -e ./log/log-%J-%I.err
# -- end of LSF options --

### 
source /work3/xenoka/miniconda3/bin/activate TorchMeta

python run_PINN.py --checkpoint_dir ./SIREN_hlmhltz --siren True --standardize_data False --loss_fn mae --lambda_bc 0 --lambda_pde 0.1 --freq_batch 105 --n_pde_samples 15000 --max_f_counter 150000 --n_hidden_layers 3 --curriculum_training False --map_input False
