#!/bin/bash
#$ -N NN_training_PC
#$ -S /bin/bash
#$ -cwd
#$ -l m_mem_free=8G

module purge

echo "_START_$(date)"

# usage ./train_NN_pc.sh [data] [act_func] [l_rate] [DS_its] [epochs] [lambda] [integration_step]

matlab -nodisplay -nosplash -nodesktop -r "run('main.m');exit;"

echo "_EXITSTAT_$?"
echo "_END_$(date)"

exit
