#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=deberta
#SBATCH --account=eecs595w24_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16gb
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=deberta.out

# The application(s) to execute along with its input arguments and options:

/bin/hostname
echo "job started"
nvidia-smi
python3 deberta.py