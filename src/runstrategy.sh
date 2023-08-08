#python model_train.py --model LeNet --dataset mnist \
#      --epochs=8 \
#      --n_calls=15 \
#      --k_fold=5 \
#      --bayesian_search \
#      --strategy='{"optimizer": "sgd", "compression": "naturalcompression", "learning_rate": 0.01}'
#
#python model_train.py --model LeNet --dataset mnist \
#      --epochs=12 \
#      --n_calls=10 \
#      --k_fold=5 \
#      --bayesian_search \
#      --strategy='{"optimizer": "sgd", "compression": "onebitsgd", "learning_rate": 0.01}'
#
#python model_train.py --model LeNet --dataset mnist \
#      --epochs=8 \
#      --n_calls=15 \
#      --k_fold=5 \
#      --bayesian_search \
#      --strategy='{"optimizer": "sgd", "compression": "sparsegradient", "learning_rate": 0.01, "drop_rate": 0.9}'
#
#python model_train.py --model LeNet --dataset mnist \
#      --epochs=8 \
#      --n_calls=15 \
#      --k_fold=5 \
#      --bayesian_search \
#      --strategy='{"optimizer": "sgd", "compression": "terngrad", "learning_rate": 0.01, "clip": 2.5}'
#
#python model_train.py --model LeNet --dataset mnist \
#      --epochs=8 \
#      --n_calls=15 \
#      --k_fold=5 \
#      --bayesian_search \
#      --strategy='{"optimizer": "sgd", "compression": "topk", "learning_rate": 0.01, "k": 10}'
#
#python model_train.py --model LeNet --dataset mnist \
#      --epochs=8 \
#      --n_calls=15 \
#      --k_fold=5 \
#      --bayesian_search \
#      --strategy='{"optimizer": "efsignsgd", "compression": "none", "learning_rate": 0.01}'
#
#python model_train.py --model LeNet --dataset mnist \
#      --epochs=8 \
#      --n_calls=15 \
#      --k_fold=5 \
#      --bayesian_search \
#      --strategy='{"optimizer": "fetchsgd", "compression": "none", "learning_rate": 0.01, "c": 10000, "r": 1}'
#
#python model_train.py --model LeNet --dataset mnist \
#      --epochs=8 \
#      --n_calls=15 \
#      --k_fold=5 \
#      --bayesian_search \
#      --strategy='{"optimizer": "memsgd", "compression": "none", "learning_rate": 0.01, "top_k": 10}'
#
#python model_train.py --model LeNet --dataset mnist \
#      --epochs=8 \
#      --n_calls=15 \
#      --k_fold=5 \
#      --bayesian_search \
#      --strategy='{"optimizer": "sgd", "compression": "gradientsparsification", "k": 0.02, "max_iter": 2, "learning_rate": 0.01}'

#python model_train.py --model LeNet --dataset mnist \
#      --epochs=250 \
#      --n_calls=10 \
#      --k_fold=5 \
#      --fullset=10 \
#      --bayesian_search \
#      --log=1 \
#      --strategy='{"optimizer": "sgd", "compression": "none", "learning_rate": 0.01}'

#python model_train.py --model LeNet --dataset mnist \
#      --epochs=20 \
#      --n_calls=10 \
#      --k_fold=5 \
#      --fullset=10 \
#      --stop_patience=25 \
#      --lambda_l2=0.001 \
#      --log=1 \
#      --strategy='{"optimizer": "sgd", "compression": "none", "learning_rate": 0.01}'
#
#python model_train.py --model LeNet --dataset mnist \
#      --epochs=20 \
#      --n_calls=10 \
#      --k_fold=5 \
#      --fullset=10 \
#      --stop_patience=25 \
#      --lambda_l2=0.001 \
#      --log=1 \
#      --strategy='{"optimizer": "sgd", "compression": "topk", "learning_rate": 0.01, "k": 5}'
#
#python model_train.py --model LeNet --dataset mnist \
#      --epochs=20 \
#      --n_calls=10 \
#      --k_fold=5 \
#      --fullset=10 \
#      --stop_patience=25 \
#      --lambda_l2=0.001 \
#      --log=1 \
#      --strategy='{"optimizer": "sgd", "compression": "topk", "learning_rate": 0.01, "k": 10}'

#python model_train.py --model LeNet --dataset mnist \
#      --epochs=200 \
#      --n_calls=10 \
#      --k_fold=5 \
#      --fullset=1 \
#      --stop_patience=15 \
#      --bayesian_search \
#      --log=1 \
#      --strategy='{"optimizer": "fetchsgd", "compression": "None", "learning_rate": 0.01, "c": 10000, "r": 1 , "momentum": 0.9}'


python model_train.py --model LeNet --dataset mnist \
      --epochs=200 \
      --n_calls=10 \
      --k_fold=5 \
      --fullset=1 \
      --stop_patience=15 \
      --bayesian_search \
      --log=1 \
      --strategy='{"optimizer": "sgd", "compression": "sparsegradient", "learning_rate": 0.1, "drop_rate": 90}'

#python model_train.py --model LeNet --dataset mnist \
#      --epochs=200 \
#      --n_calls=10 \
#      --k_fold=5 \
#      --fullset=1 \
#      --stop_patience=15 \
#      --bayesian_search \
#      --log=1 \
#      --strategy='{"optimizer": "efsignsgd", "compression": "None", "learning_rate": 0.01}'


#python model_train.py --model LeNet --dataset mnist \
#      --epochs=250 \
#      --n_calls=10 \
#      --k_fold=5 \
#      --fullset=10 \
#      --bayesian_search \
#      --log=1 \
#      --strategy='{"optimizer": "sgd", "compression": "naturalcompression", "learning_rate": 0.01}'
#
#python model_train.py --model LeNet --dataset mnist \
#      --epochs=250 \
#      --n_calls=10 \
#      --k_fold=5 \
#      --fullset=10 \
#      --bayesian_search \
#      --log=1 \
#      --strategy='{"optimizer": "sgd", "compression": "onebitsgd", "learning_rate": 0.01}'