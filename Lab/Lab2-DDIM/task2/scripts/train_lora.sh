export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="imio/anime-face-lora"
export OUTPUT_DIR="./runs/Genshin-model-lora-2"

accelerate launch --mixed_precision="no" train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --dataset_name=$DATASET_NAME \
  --caption_column="text" \
  --resolution=512 \
  --random_flip \
  --train_batch_size=2 \
  --num_train_epochs=150 \
  --checkpointing_steps=500 \
  --learning_rate=5e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --rank=32 \
  --seed=42 \
  --checkpoints_total_limit=2 \
  --validation_prompt="a girl wearing glasses"

  