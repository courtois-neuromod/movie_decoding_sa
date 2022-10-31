#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --job-name=Frineds_vqvae
#SBATCH --time=11:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40
#SBATCH --mem=500G
#SBATCH --output "out/Frineds_VQVAE_%A.out"

export NCCL_BLOCKING_WAIT=1

python scripts/train_vqvae.py --precision 32 \
			      --dataset friends \
			      --data_path "home/sana4471/projects/rrg-pbellec/sana4471/movie_decoding_sa/restart2/H5/Total_h5py.hdf5" \
			      --max_epochs 150 \
			      --batch_size 32 \
			      --gpus 4 \
			      --num_workers 16 \
			      --default_root_dir "/project/rrg-pbellec/sana4471/movie_decoding_sa/restart2/H5/video_transformer-main/video_transformer-main/model" \
			      --n_codes 1024 \
			      --embedding_dim 128 \
			      --n_res_layers 4 \
			      --downsample 4 16 16 \
			      --sequence_length 16 \
			      --resolution 128 \
			      --gradient_clip_val 1 \
			      --sync_batchnorm

