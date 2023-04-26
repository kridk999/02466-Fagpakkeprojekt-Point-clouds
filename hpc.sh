#!/bin/sh
### General options
### –- specify queue --
# BSUB -q gpuv100
### -- set the job Name --
# BSUB -J vfnet
### -- ask for number of cores (default: 1) --
# BSUB -n 16
### -- Select the resources: 1 gpu in exclusive process mode --
# BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
# BSUB -W 23:59
# request 5GB of system-memory
# BSUB -R "rusage[mem=16GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
# BSUB -u ziruoye@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
# BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
# BSUB -o gpu_%J.out
# BSUB -e gpu_%J.err
# -- end of LSF options --
​
nvidia-smi
# Load the cuda module
module load python3/3.9.5 cuda/11.4 cudnn/v8.2.2.26-prod-cuda-11.4
source venv/bin/activate
​
python3 ./main.py --dataset teeth --x_train /work3/ziruoye/data/publication_v2.0.0_train_small --x_test /work3/ziruoye/data/publication_v2.0.0_test_small \
--batch_size 128 --num_epochs 15000 --num_workers 16 --exp_name 512_latent_encoder_dim_fix_large_std --model vae --feat_dims 1024 --resume_from /work3/ziruoye/vfnet/runs/b6adb79/512_latent_encoder_dim_fix_large_std/epoch_5000.pth.tar