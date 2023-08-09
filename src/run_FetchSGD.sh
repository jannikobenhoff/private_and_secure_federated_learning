# Hyperparameter Search
#python model_train.py --model LeNet --dataset mnist \
#      --epochs=200 \
#      --n_calls=10 \
#      --k_fold=5 \
#      --fullset=1 \
#      --stop_patience=15 \
#      --bayesian_search \
#      --log=2 \
#      --strategy='{"optimizer": "fetchsgd", "compression": "none", "learning_rate": 0.1, "c": 100, "r": 1, "momentum": 0.9}'
#
#python model_train.py --model LeNet --dataset mnist \
#      --epochs=200 \
#      --n_calls=10 \
#      --k_fold=5 \
#      --fullset=1 \
#      --stop_patience=15 \
#      --bayesian_search \
#      --log=2 \
#      --strategy='{"optimizer": "fetchsgd", "compression": "none", "learning_rate": 0.1, "c": 1000, "r": 1, "momentum": 0.9}'

# Training
python model_train.py --model LeNet --dataset mnist \
      --epochs=20 \
      --k_fold=1 \
      --fullset=100 \
      --stop_patience=20 \
      --lr_decay=5 \
      --log=2 \
      --strategy='{"optimizer": "fetchsgd", "compression": "none", "learning_rate": 0.01, "c": 100, "r": 1, "momentum": 0.9}'

python model_train.py --model LeNet --dataset mnist \
      --epochs=20 \
      --k_fold=1 \
      --fullset=100 \
      --stop_patience=20 \
      --lr_decay=5 \
      --log=2 \
      --strategy='{"optimizer": "fetchsgd", "compression": "none", "learning_rate": 0.01, "c": 1000, "r": 1, "momentum": 0.9}'

python model_train.py --model LeNet --dataset mnist \
      --epochs=20 \
      --k_fold=1 \
      --fullset=100 \
      --stop_patience=20 \
      --lr_decay=5 \
      --log=2 \
      --strategy='{"optimizer": "fetchsgd", "compression": "none", "learning_rate": 0.01, "c": 10000, "r": 1, "momentum": 0.9}'