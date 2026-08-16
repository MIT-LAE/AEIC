#!/bin/bash

#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000MB
#SBATCH --hint=nomultithread
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --mail-user=iross@mit.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --job-name=aeic-t2g-map
#SBATCH --output=log/t2g-map-%j.out

# This script is designed to run as a job array based on the camera-list.txt
# file. It should be submitted with a command like:
#
# sbatch --array=1-100 t2g-era-map.sh

if [[ -z $SLURM_ARRAY_JOB_ID ]]
then
    echo NOT RUNNING AS A JOB ARRAY!!!
    exit 1
fi

PROJ_BASE=/home/iross/code/AEIC
RUN_BASE=/home/iross/data/AEIC
RUN=$RUN_BASE/oag-2019
IN=$RUN/inputs
OUT=$RUN/map/slice

uv --project $PROJ_BASE run aeic trajectories-to-grid --mode map \
  --input-store $RUN/oag-2019.aeic-store \
  --mission-db-file $RUN_BASE/oag-2019.sqlite \
  --grid-file $RUN/era5-1x1-grid.toml \
  --map-prefix $OUT \
  --slice-count $SLURM_ARRAY_TASK_COUNT \
  --slice-index $((SLURM_ARRAY_TASK_ID - 1))
