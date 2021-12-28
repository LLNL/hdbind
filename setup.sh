ml load gcc/10.2.1
ml load cuda/11.1.0
export CUDA_PATH=/usr/tce/packages/cuda/cuda-11.1.0
# below you need to specify the compute capability for the CUDA device
export SMS=60
conda activate HD_env
export PYTHONPATH=$PWD/hdpy:$PYTHONPATH
