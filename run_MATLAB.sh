#!/bin/bash
#$ -N MATLAB_SCRIPT
#$ -S /bin/bash
#$ -cwd
#$ -l m_mem_free=32G

module purge

echo "_START_$(date)"

# usage ./run_MATLAB.sh [MATLAB_script.m]
#matlab -nodisplay -nosplash -nodesktop -r "run('${1}');exit;"

# usage ./run_MATLAB.sh [one-line MATLAB command]
echo "running "$@""
matlab -nodisplay -nosplash -nodesktop -r "$@"

echo "_EXITSTAT_$?"
echo "_END_$(date)"

exit
