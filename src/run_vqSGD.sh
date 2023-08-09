# Hyperparameter Search
#python model_train.py --model LeNet --dataset mnist \
#      --epochs=200 \
#      --n_calls=10 \
#      --k_fold=5 \
#      --fullset=1 \
#      --stop_patience=15 \
#      --bayesian_search \
#      --log=1 \
#      --strategy='{"optimizer": "sgd", "compression": "vqsgd", "learning_rate": 0.001, "repetition": 5}'
#
#python model_train.py --model LeNet --dataset mnist \
#      --epochs=200 \
#      --n_calls=10 \
#      --k_fold=5 \
#      --fullset=1 \
#      --stop_patience=15 \
#      --bayesian_search \
#      --log=1 \
#      --strategy='{"optimizer": "sgd", "compression": "vqsgd", "learning_rate": 0.001, "repetition": 10}'
#
#python model_train.py --model LeNet --dataset mnist \
#      --epochs=200 \
#      --n_calls=10 \
#      --k_fold=5 \
#      --fullset=1 \
#      --stop_patience=15 \
#      --bayesian_search \
#      --log=1 \
#      --strategy='{"optimizer": "sgd", "compression": "vqsgd", "learning_rate": 0.01, "repetition": 50}'
#
#python model_train.py --model LeNet --dataset mnist \
#      --epochs=200 \
#      --n_calls=10 \
#      --k_fold=5 \
#      --fullset=1 \
#      --stop_patience=15 \
#      --bayesian_search \
#      --log=1 \
#      --strategy='{"optimizer": "sgd", "compression": "vqsgd", "learning_rate": 0.1, "repetition": 100}'
#
#python model_train.py --model LeNet --dataset mnist \
#      --epochs=200 \
#      --n_calls=10 \
#      --k_fold=5 \
#      --fullset=1 \
#      --stop_patience=15 \
#      --bayesian_search \
#      --log=1 \
#      --strategy='{"optimizer": "sgd", "compression": "vqsgd", "learning_rate": 0.1, "repetition": 200}'

#python model_train.py --model LeNet --dataset mnist \
#      --epochs=200 \
#      --n_calls=10 \
#      --k_fold=5 \
#      --fullset=1 \
#      --stop_patience=15 \
#      --bayesian_search \
#      --log=1 \
#      --strategy='{"optimizer": "sgd", "compression": "vqsgd", "learning_rate": 0.01, "repetition": 500}'

#Training
#python model_train.py --model LeNet --dataset mnist \
#      --epochs=20 \
#      --k_fold=1 \
#      --fullset=100 \
#      --stop_patience=20 \
#      --lr_decay=5 \
#      --log=1 \
#      --strategy='{"optimizer": "sgd", "compression": "vqsgd", "learning_rate": 0.001, "repetition": 5}'

python model_train.py --model LeNet --dataset mnist \
      --epochs=20 \
      --k_fold=1 \
      --fullset=100 \
      --stop_patience=20 \
      --lr_decay=5 \
      --log=2 \
      --strategy='{"optimizer": "sgd", "compression": "vqsgd", "learning_rate": 0.001, "repetition": 10}'


python model_train.py --model LeNet --dataset mnist \
      --epochs=20 \
      --k_fold=1 \
      --fullset=100 \
      --stop_patience=20 \
      --lr_decay=5 \
      --log=2 \
      --strategy='{"optimizer": "sgd", "compression": "vqsgd", "learning_rate": 0.01, "repetition": 50}'


python model_train.py --model LeNet --dataset mnist \
      --epochs=20 \
      --k_fold=1 \
      --fullset=100 \
      --stop_patience=20 \
      --lr_decay=5 \
      --log=2 \
      --strategy='{"optimizer": "sgd", "compression": "vqsgd", "learning_rate": 0.01, "repetition": 100}'


python model_train.py --model LeNet --dataset mnist \
      --epochs=20 \
      --k_fold=1 \
      --fullset=100 \
      --stop_patience=20 \
      --log=2 \
      --strategy='{"optimizer": "sgd", "compression": "vqsgd", "learning_rate": 0.01, "repetition": 200}'


python model_train.py --model LeNet --dataset mnist \
      --epochs=20 \
      --k_fold=1 \
      --fullset=100 \
      --stop_patience=20 \
      --lr_decay=5 \
      --log=2 \
      --strategy='{"optimizer": "sgd", "compression": "vqsgd", "learning_rate": 0.01, "repetition": 500}'


