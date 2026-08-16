#!/bin/bash

#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000MB
#SBATCH --hint=nomultithread
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --mail-user=iross@mit.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --job-name=aeic-test
#SBATCH --output=log/test-%j.out

PROJ_BASE=/home/iross/code/AEIC
RUN_BASE=/home/iross/data/AEIC
RUN=$RUN_BASE/oag-2019
IN=$RUN/inputs
OUT=$RUN/test-run/slice

uv --project $PROJ_BASE run aeic run \
  --config-file $IN/config.toml \
  --performance-selector-dir $IN/performance \
  --mission-db-file $RUN_BASE/oag-2019.sqlite \
  --output-store $OUT \
  --slice-count 1000 \
  --slice-index 0
