#!/bin/bash
#
#PBS -N skopt-tune
#PBS -l select=1:ncpus=4:mem=200gb:ngpus=1:interconnect=1g:gpu_model=rtx6000
#PBS -l walltime=47:59:00
#PBS -o output.txt
#PBS -j oe
#PBS -q skygpu


source /home/qianxiangai/.bashrc

module load anaconda3/2021.05-gcc/8.3.1
module load cuda/10.2.89-gcc/8.3.1

module list

conda activate CompCycleGAN
which python
export PYTHONPATH="${PYTHONPATH}:/scratch1/qianxiangai/augcyc/"

echo $CUDA_HOME

cd ${PBS_O_WORKDIR} 
echo $PWD
python tune.py


