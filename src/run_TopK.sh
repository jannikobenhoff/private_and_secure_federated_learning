# Hyperparameter Search
python model_train.py --model LeNet --dataset mnist \
      --epochs=200 \
      --n_calls=10 \
      --k_fold=5 \
      --fullset=1 \
      --stop_patience=15 \
      --bayesian_search \
      --log=1 \
      --strategy='{"optimizer": "sgd", "compression": "topk", "learning_rate": 0.1, "k": 5}'

python model_train.py --model LeNet --dataset mnist \
      --epochs=200 \
      --n_calls=10 \
      --k_fold=5 \
      --fullset=1 \
      --stop_patience=15 \
      --bayesian_search \
      --log=1 \
      --strategy='{"optimizer": "sgd", "compression": "topk", "learning_rate": 0.1, "k": 20}'

# Training
#python model_train.py --model LeNet --dataset mnist \
#      --epochs=20 \
#      --k_fold=1 \
#      --fullset=100 \
#      --stop_patience=20 \
#      --lambda_l2=0.0075632494949532695 \
#      --log=1 \
#      --strategy='{"optimizer": "sgd", "compression": "topk", "learning_rate": 0.01, "k": 2}'
#
#python model_train.py --model LeNet --dataset mnist \
#      --epochs=20 \
#      --k_fold=1 \
#      --fullset=100 \
#      --lambda_l2=0.0075632494949532695 \
#      --stop_patience=20 \
#      --log=1 \
#      --strategy='{"optimizer": "sgd", "compression": "topk", "learning_rate": 0.01, "k": 3}'
#
#python model_train.py --model LeNet --dataset mnist \
#      --epochs=20 \
#      --k_fold=1 \
#      --fullset=100 \
#      --lambda_l2=0.0075632494949532695 \
#      --stop_patience=20 \
#      --log=1 \
#      --strategy='{"optimizer": "sgd", "compression": "topk", "learning_rate": 0.01, "k": 5}'
#
#python model_train.py --model LeNet --dataset mnist \
#      --epochs=20 \
#      --k_fold=1 \
#      --fullset=100 \
#      --lambda_l2=0.0075632494949532695 \
#      --stop_patience=20 \
#      --log=1 \
#      --strategy='{"optimizer": "sgd", "compression": "topk", "learning_rate": 0.01, "k": 7}'