python scripts/05_train_encoder.py \
    cesm_path=/scratch/local/aygh9582/CESM2/monthly \
    checkpoint_dir=/scratch/alpine/aygh9582/checkpoints \
    model_name=earthae_wide_1 \
    num_workers=48 \
    batch_size=3 \
    resume_step=7000 \
    resume_checkpoint=/scratch/alpine/aygh9582/checkpoints/earthae_wide_1_step_7000.ckpt
