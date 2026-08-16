#!/bin/bash

#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000MB
#SBATCH --hint=nomultithread
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --mail-user=iross@mit.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --job-name=aeic-merge
#SBATCH --output=log/merge-%j.out

PROJ_BASE=/home/iross/code/AEIC
RUN_BASE=/home/iross/data/AEIC
RUN=$RUN_BASE/oag-2019
IN=$RUN/inputs
OUT=$RUN/run/slice

uv --project $PROJ_BASE run aeic merge-stores \
  --output-store $RUN/oag-2019.aeic-store \
  --merge \
  ${OUT}*.nc
