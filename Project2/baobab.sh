#!/bin/bash
#SBATCH --cpus-per-task=5
#SBATCH --ntasks=1
#SBATCH --licenses=matlab@matlablm.unige.ch
#SBATCH --partition=shared-bigmem
#SBATCH --time=03:15:00
#SBATCH --mail-user=monika.avila@unige.ch
#SBATCH --mail-type=END
#SBATCH --mem=128G

module load foss/2016b Python/3.5.2

BASE_MFILE_NAME=Analysis_Predicted_Coef

unset DISPLAY

echo "Running ${BASE_MFILE_NAME}.m on $(hostname)"

srun -nodesktop -nosplash -nodisplay -r ${BASE_MFILE_NAME}
