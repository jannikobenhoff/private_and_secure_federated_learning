# Hyperparameter Search
#python model_train.py --model LeNet --dataset mnist \
#      --epochs=200 \
#      --n_calls=10 \
#      --k_fold=5 \
#      --fullset=1 \
#      --stop_patience=15 \
#      --bayesian_search \
#      --log=1 \
#      --strategy='{"optimizer": "memsgd", "compression": "none", "learning_rate": 0.1, "top_k": 5, "rand_k": "None"}'

#python model_train.py --model LeNet --dataset mnist \
#      --epochs=200 \
#      --n_calls=10 \
#      --k_fold=5 \
#      --fullset=1 \
#      --stop_patience=15 \
#      --bayesian_search \
#      --log=1 \
#      --strategy='{"optimizer": "memsgd", "compression": "none", "learning_rate": 0.1, "top_k": 10, "rand_k": "None"}'

#python model_train.py --model LeNet --dataset mnist \
#      --epochs=200 \
#      --n_calls=10 \
#      --k_fold=5 \
#      --fullset=1 \
#      --stop_patience=15 \
#      --bayesian_search \
#      --log=1 \
#      --strategy='{"optimizer": "memsgd", "compression": "none", "learning_rate": 0.1, "top_k": 50, "rand_k": "None"}'

#python model_train.py --model LeNet --dataset mnist \
#      --epochs=200 \
#      --n_calls=10 \
#      --k_fold=5 \
#      --fullset=1 \
#      --stop_patience=15 \
#      --bayesian_search \
#      --log=1 \
#      --strategy='{"optimizer": "memsgd", "compression": "none", "learning_rate": 0.1, "top_k": 100, "rand_k": "None"}'

# Training
python model_train.py --model LeNet --dataset mnist \
      --epochs=20 \
      --k_fold=1 \
      --fullset=100 \
      --stop_patience=20 \
      --lr_decay=5 \
      --log=2 \
      --strategy='{"optimizer": "memsgd", "compression": "none", "learning_rate": 0.001, "top_k": 5, "rand_k": "None"}'

python model_train.py --model LeNet --dataset mnist \
      --epochs=20 \
      --k_fold=1 \
      --fullset=100 \
      --stop_patience=20 \
      --lr_decay=5 \
      --log=2 \
      --strategy='{"optimizer": "memsgd", "compression": "none", "learning_rate": 0.01, "top_k": 10, "rand_k": "None"}'

python model_train.py --model LeNet --dataset mnist \
      --epochs=20 \
      --k_fold=1 \
      --fullset=100 \
      --stop_patience=20 \
      --lr_decay=5 \
      --log=2 \
      --strategy='{"optimizer": "memsgd", "compression": "none", "learning_rate": 0.01, "top_k": 50, "rand_k": "None"}'

python model_train.py --model LeNet --dataset mnist \
      --epochs=20 \
      --k_fold=1 \
      --fullset=100 \
      --stop_patience=20 \
      --lr_decay=5 \
      --log=2 \
      --strategy='{"optimizer": "memsgd", "compression": "none", "learning_rate": 0.01, "top_k": 100, "rand_k": "None"}'
