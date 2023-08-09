# Hyperparameter Search
python model_train.py --model LeNet --dataset mnist \
      --epochs=200 \
      --n_calls=10 \
      --k_fold=5 \
      --fullset=1 \
      --stop_patience=15 \
      --bayesian_search \
      --log=1 \
      --strategy='{"optimizer": "sgd", "compression": "sparsegradient", "learning_rate": 0.01, "drop_rate": 99}'

python model_train.py --model LeNet --dataset mnist \
      --epochs=200 \
      --n_calls=10 \
      --k_fold=5 \
      --fullset=1 \
      --stop_patience=15 \
      --bayesian_search \
      --log=1 \
      --strategy='{"optimizer": "sgd", "compression": "sparsegradient", "learning_rate": 0.01, "drop_rate": 95}'

python model_train.py --model LeNet --dataset mnist \
      --epochs=200 \
      --n_calls=10 \
      --k_fold=5 \
      --fullset=1 \
      --stop_patience=15 \
      --bayesian_search \
      --log=1 \
      --strategy='{"optimizer": "sgd", "compression": "sparsegradient", "learning_rate": 0.01, "drop_rate": 85}'

python model_train.py --model LeNet --dataset mnist \
      --epochs=200 \
      --n_calls=10 \
      --k_fold=5 \
      --fullset=1 \
      --stop_patience=15 \
      --bayesian_search \
      --log=1 \
      --strategy='{"optimizer": "sgd", "compression": "sparsegradient", "learning_rate": 0.01, "drop_rate": 80}'


#Training
#python model_train.py --model LeNet --dataset mnist \
#      --epochs=20 \
#      --k_fold=1 \
#      --fullset=100 \
#      --stop_patience=20 \
#      --lambda_l2=a \
#      --log=1 \
#      --strategy='{"optimizer": "sgd", "compression": "sparsegradient", "learning_rate": 0.1, "drop_rate": 90}'
