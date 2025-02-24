#!/bin/bash

help="Usage: run.sh <script> [args...]"

PYTHON_ENV=/home/marchiorot/miniforge3/envs/ransomwise/bin/python

SCRIPT_PATH=$1


# Check if the script exists
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "Invalid script path: ${SCRIPT_PATH}"
    echo "${help}"
    exit 1
fi

# Use all other args as arguments to the script
ARGS=${@:2}

# Replace / with .
SCRIPT_MODULE=${SCRIPT_PATH//\//.}
# Remove the .py suffix
SCRIPT_MODULE=${SCRIPT_MODULE%.py}
SCRIPT_CMD="${PYTHON_ENV} -m ${SCRIPT_MODULE} ${ARGS}"

echo "Running ${SCRIPT_CMD}"
bash -c "${SCRIPT_CMD}"