python model_train.py --model LeNet --dataset mnist \
      --epochs=8 \
      --n_calls=15 \
      --k_fold=5 \
      --bayesian_search \
      --strategy='{"optimizer": "sgd", "compression": "naturalcompression", "learning_rate": 0.01}'
#
#python model_train.py --model LeNet --dataset mnist \
#      --epochs=12 \
#      --n_calls=10 \
#      --k_fold=5 \
#      --bayesian_search \
#      --strategy='{"optimizer": "sgd", "compression": "onebitsgd", "learning_rate": 0.01}'
#
python model_train.py --model LeNet --dataset mnist \
      --epochs=8 \
      --n_calls=15 \
      --k_fold=5 \
      --bayesian_search \
      --strategy='{"optimizer": "sgd", "compression": "sparsegradient", "learning_rate": 0.01, "drop_rate": 0.9}'

python model_train.py --model LeNet --dataset mnist \
      --epochs=8 \
      --n_calls=15 \
      --k_fold=5 \
      --bayesian_search \
      --strategy='{"optimizer": "sgd", "compression": "terngrad", "learning_rate": 0.01, "clip": 2.5}'

python model_train.py --model LeNet --dataset mnist \
      --epochs=8 \
      --n_calls=15 \
      --k_fold=5 \
      --bayesian_search \
      --strategy='{"optimizer": "sgd", "compression": "topk", "learning_rate": 0.01, "k": 10}'

python model_train.py --model LeNet --dataset mnist \
      --epochs=8 \
      --n_calls=15 \
      --k_fold=5 \
      --bayesian_search \
      --strategy='{"optimizer": "efsignsgd", "compression": "none", "learning_rate": 0.01}'

python model_train.py --model LeNet --dataset mnist \
      --epochs=8 \
      --n_calls=15 \
      --k_fold=5 \
      --bayesian_search \
      --strategy='{"optimizer": "fetchsgd", "compression": "none", "learning_rate": 0.01, "c": 10000, "r": 1}'

python model_train.py --model LeNet --dataset mnist \
      --epochs=8 \
      --n_calls=15 \
      --k_fold=5 \
      --bayesian_search \
      --strategy='{"optimizer": "memsgd", "compression": "none", "learning_rate": 0.01, "top_k": 10}'

python model_train.py --model LeNet --dataset mnist \
      --epochs=8 \
      --n_calls=15 \
      --k_fold=5 \
      --bayesian_search \
      --strategy='{"optimizer": "sgd", "compression": "gradientsparsification", "k": 0.02, "max_iter": 2, "learning_rate": 0.01}'

python model_train.py --model LeNet --dataset mnist \
      --epochs=8 \
      --n_calls=15 \
      --k_fold=5 \
      --bayesian_search \
      --strategy='{"optimizer": "sgd", "compression": "none", "learning_rate": 0.01}'