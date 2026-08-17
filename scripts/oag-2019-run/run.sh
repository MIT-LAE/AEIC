#!/bin/bash

#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000MB
#SBATCH --hint=nomultithread
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --mail-user=iross@mit.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --job-name=aeic-run
#SBATCH --output=log/run-%j.out

# This script is designed to run as a job array based on the camera-list.txt
# file. It should be submitted with a command like:
#
# sbatch --array=1-100 run.sh

if [[ -z $SLURM_ARRAY_JOB_ID ]]
then
    echo NOT RUNNING AS A JOB ARRAY!!!
    exit 1
fi

export PYTHONUNBUFFERED=1

PROJ_BASE=/home/iross/code/AEIC
RUN_BASE=/home/iross/data/AEIC
RUN=$RUN_BASE/oag-2019
IN_SRC=$RUN/inputs
OUT=$RUN/run/slice
JOB_TMP=/tmp/AEIC-${SLURM_ARRAY_TASK_ID}
IN=${JOB_TMP}/inputs

function cleanup {
  rm -fr ${JOB_TMP}
}
trap cleanup EXIT

rm -fr ${JOB_TMP}
mkdir -p ${JOB_TMP}
cp $RUN_BASE/oag-2019.sqlite ${JOB_TMP}/oag-2019.sqlite
cp -r $IN_SRC ${JOB_TMP}

sed -i "s|${IN_SRC}|${IN}|g" $IN/config.toml

uv --project $PROJ_BASE run aeic run \
  --config-file $IN/config.toml \
  --performance-selector-dir $IN/performance \
  --mission-db-file ${JOB_TMP}/oag-2019.sqlite \
  --output-store $OUT \
  --slice-count $SLURM_ARRAY_TASK_COUNT \
  --slice-index $((SLURM_ARRAY_TASK_ID - 1))
