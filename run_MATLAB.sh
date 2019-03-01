#!/bin/bash
#$ -N MATLAB_SCRIPT
#$ -S /bin/bash
#$ -cwd
#$ -l m_mem_free=32G

module purge

echo "_START_$(date)"

# usage ./run_MATLAB.sh [MATLAB_script.m]

matlab -nodisplay -nosplash -nodesktop -r "run('${1}');exit;"

echo "_EXITSTAT_$?"
echo "_END_$(date)"

exit
