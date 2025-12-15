python scripts/05_train_encoder.py \
    cesm_path=/scratch/local/aygh9582/CESM2/monthly \
    checkpoint_dir=/scratch/alpine/aygh9582/checkpoints \
    model_name=earthae_wide_5 \
    num_workers=48 \
    batch_size=13 \
    grad_accumulation_steps=8 \
    base_lr=0.00025 \
    epochs=10 \
    prefetch_factor=4 \
    pin_memory=false \
    resume_step=0 \
    warmup_steps=120 \
    rotation_p=0.5 \
    rotation_lows_deg="[-20,-30]" \
    rotation_highs_deg="[20,30]" \
    resume_checkpoint=/scratch/alpine/aygh9582/checkpoints/earthae_wide_4_step_30500.ckpt
