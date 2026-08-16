#!/bin/bash

#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000MB
#SBATCH --hint=nomultithread
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --mail-user=iross@mit.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --job-name=aeic-t2g-reduce
#SBATCH --output=log/t2g-reduce-%j.out

PROJ_BASE=/home/iross/code/AEIC
RUN_BASE=/home/iross/data/AEIC
RUN=$RUN_BASE/oag-2019
IN=$RUN/inputs
OUT=$RUN/map/slice

uv --project $PROJ_BASE run aeic trajectories-to-grid --mode reduce \
  --input-store $RUN/oag-2019.aeic-store \
  --mission-db-file $RUN_BASE/oag-2019.sqlite \
  --grid-file $RUN/era5-1x1-grid.toml \
  --map-prefix $OUT \
  --output-file $RUN/oag-2019.nc
