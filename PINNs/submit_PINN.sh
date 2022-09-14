#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J PINN
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
# request 32GB of system-memory
#BSUB -R "rusage[mem=24GB]"
### -- send notification at start -- 
### -- send notification at completion -- 
#BSUB -N 
#BSUB -o ./log/log-%J-%I.out
#BSUB -e ./log/log-%J-%I.err
# -- end of LSF options --

### 
source /work3/xenoka/miniconda3/bin/activate DSC_Project

python run_PINN.py --checkpoint_dir "./Siren_w_bcs" --siren True