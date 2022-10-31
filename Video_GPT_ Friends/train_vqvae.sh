#SBATCH --time=33:56:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=498G
#SBATCH --output "slurm/out/shinobi_VQVAE_%A.out"

module load cuda/11
module load python/3.7.7
source env_torch_nrvl/bin/activate

mkdir "$SLURM_TMPDIR/data"
cp data/shinobi_frames_128.hdf5 $SLURM_TMPDIR/data/shinobi_frames.hdf5

export NCCL_BLOCKING_WAIT=1

python scripts/train_vqvae.py --precision 32 \
			      --dataset shinobi \
			      --data_path "$SLURM_TMPDIR/data/shinobi_frames.hdf5" \
			      --max_epochs 150 \
			      --batch_size 32 \
			      --gpus 4 \
			      --num_workers 16 \
			      --default_root_dir models/shinobi_VQVAE \
			      --n_codes 1024 \
			      --embedding_dim 128 \
			      --n_res_layers 4 \
			      --downsample 4 16 16 \
			      --levels all \
			      --sequence_length 16 \
			      --resolution 128 \
			      --gradient_clip_val 1 \
			      --sync_batchnorm

