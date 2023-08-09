# Hyperparameter Search

#Training
python model_train.py --model LeNet --dataset mnist \
      --epochs=20 \
      --k_fold=1 \
      --fullset=100 \
      --stop_patience=20 \
      --lr_decay=5 \
      --log=2 \
      --strategy='{"optimizer": "sgd", "compression": "naturalcompression", "learning_rate": 0.01}'
