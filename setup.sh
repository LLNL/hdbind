#ml load gcc/10.2.1
ml load gcc/10.3.1
ml load cuda/11.1.0
export CUDA_PATH=/usr/tce/packages/cuda/cuda-11.1.0
export LD_LIBRARY_PATH=/usr/workspace/wsa/jones289/miniconda3/envs/HD_env/lib:$LD_LIBRARY_PATH
# below you need to specify the compute capability for the CUDA device
export SMS=60
conda activate /usr/workspace/wsa/jones289/miniconda3/envs/HD_env 
export PYTHONPATH=$PWD/hdpy:$PYTHONPATH
