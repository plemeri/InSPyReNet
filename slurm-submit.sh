#!/bin/bash

## JOB NAME
#SBATCH --job-name=inspyrenet
## Name of stdout output file (%j expands to %jobId)
#SBATCH --output=train.%j.out
## Queue name or partiton name (2080ti, titanxp, titanrtx)
#SBATCH --partition=A5000
#SBATCH --time=24:00:00

#SBATCH --nodes=1  # always 1
## Specifying nodelist makes your priority higher
##SBATCH --nodelist=n15
## Number of gpus
#SBATCH --gres=gpu:1
## Same as gres
#SBATCH --ntasks-per-node=1
## Number of cores per task
#SBATCH --cpus-per-task=2

#SBATCH --mail-type=ALL
#SBATCH --mail-user=taehoon1018@postech.ac.kr


cd $SLURM_SUBMIT_DIR

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR" echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

module purge  # Remove all modules.
module load postech  

echo "Start"

echo "source $HOME/anaconda3/etc/profile.d/conda.sh"
source $HOME/anaconda3/etc/profile.d/conda.sh

echo "conda activate inspyrenet" 
conda activate inspyrenet
cd Projects/InSPyReNet

python Expr.py --config $1 --verbose --debug

date

echo "conda deactivate"
conda deactivate

squeue --job $SLURM_JOBID
