python scripts/05_train_encoder.py \
    cesm_path=/scratch/local/aygh9582/CESM2/monthly \
    checkpoint_dir=/scratch/alpine/aygh9582/checkpoints \
    model_name=earthae_wide_3 \
    num_workers=48 \
    batch_size=12 \
    grad_accumulation_steps=8 \
    prefetch_factor=4 \
    resume_step=0 \
    warmup_steps=100 \
    resume_checkpoint=/scratch/alpine/aygh9582/checkpoints/earthae_wide_2_step_225000.ckpt
