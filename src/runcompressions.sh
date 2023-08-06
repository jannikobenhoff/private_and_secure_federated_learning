python model_train.py --model LeNet --dataset mnist \
      --epochs=20 \
      --k_fold=5 \
      --fullset=10 \
      --stop_patience=20 \
      --log=0 \
      --strategy='{"optimizer": "sgd", "compression": "vqsgd", "learning_rate": 0.1, "repetition": 1}'

python model_train.py --model LeNet --dataset mnist \
      --epochs=20 \
      --k_fold=5 \
      --fullset=10 \
      --stop_patience=20 \
      --log=0 \
      --strategy='{"optimizer": "sgd", "compression": "vqsgd", "learning_rate": 0.1, "repetition": 10}'

python model_train.py --model LeNet --dataset mnist \
      --epochs=20 \
      --k_fold=5 \
      --fullset=10 \
      --stop_patience=20 \
      --log=0 \
      --strategy='{"optimizer": "sgd", "compression": "vqsgd", "learning_rate": 0.1, "repetition": 100}'

python model_train.py --model LeNet --dataset mnist \
      --epochs=20 \
      --k_fold=5 \
      --fullset=10 \
      --stop_patience=20 \
      --log=0 \
      --strategy='{"optimizer": "sgd", "compression": "vqsgd", "learning_rate": 0.1, "repetition": 200}'