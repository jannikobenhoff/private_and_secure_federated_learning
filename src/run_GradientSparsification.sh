# Hyperparameter Search
python model_train.py --model LeNet --dataset mnist \
      --epochs=200 \
      --n_calls=10 \
      --k_fold=5 \
      --fullset=1 \
      --stop_patience=15 \
      --bayesian_search \
      --log=1 \
      --strategy='{"optimizer": "sgd", "compression": "gradientsparsification", "learning_rate": 0.01, "k": 0.02, "max_iter": 2}'

python model_train.py --model LeNet --dataset mnist \
      --epochs=200 \
      --n_calls=10 \
      --k_fold=5 \
      --fullset=1 \
      --stop_patience=15 \
      --bayesian_search \
      --log=1 \
      --strategy='{"optimizer": "sgd", "compression": "gradientsparsification", "learning_rate": 0.01, "k": 0.02,"max_iter": 3}'

python model_train.py --model LeNet --dataset mnist \
      --epochs=200 \
      --n_calls=10 \
      --k_fold=5 \
      --fullset=1 \
      --stop_patience=15 \
      --bayesian_search \
      --log=1 \
      --strategy='{"optimizer": "sgd", "compression": "gradientsparsification", "learning_rate": 0.01, "k": 0.02,"max_iter": 4}'

python model_train.py --model LeNet --dataset mnist \
      --epochs=200 \
      --n_calls=10 \
      --k_fold=5 \
      --fullset=1 \
      --stop_patience=15 \
      --bayesian_search \
      --log=1 \
      --strategy='{"optimizer": "sgd", "compression": "gradientsparsification", "learning_rate": 0.01, "k": 0.1,"max_iter": 2}'

python model_train.py --model LeNet --dataset mnist \
      --epochs=200 \
      --n_calls=10 \
      --k_fold=5 \
      --fullset=1 \
      --stop_patience=15 \
      --bayesian_search \
      --log=1 \
      --strategy='{"optimizer": "sgd", "compression": "gradientsparsification", "learning_rate": 0.01, "k": 0.05,"max_iter": 2}'

#Training
#python model_train.py --model LeNet --dataset mnist \
#      --epochs=20 \
#      --k_fold=1 \
#      --fullset=100 \
#      --stop_patience=20 \
#      --lambda_l2=0.0075632494949532695 \
#      --log=1 \
#      --strategy='{"optimizer": "sgd", "compression": "gradientsparsification", "learning_rate": 0.01, "k": 0.02,"max_iter": 2}'
#
#python model_train.py --model LeNet --dataset mnist \
#      --epochs=20 \
#      --k_fold=1 \
#      --fullset=100 \
#      --stop_patience=20 \
#      --lambda_l2=0.0075632494949532695 \
#      --log=1 \
#      --strategy='{"optimizer": "sgd", "compression": "gradientsparsification", "learning_rate": 0.01, "k": 0.02,"max_iter": 5}'
#
#python model_train.py --model LeNet --dataset mnist \
#      --epochs=20 \
#      --k_fold=1 \
#      --fullset=100 \
#      --stop_patience=20 \
#      --lambda_l2=0.0075632494949532695 \
#      --log=1 \
#      --strategy='{"optimizer": "sgd", "compression": "gradientsparsification", "learning_rate": 0.01, "k": 0.02,"max_iter": 10}'
#
#  python model_train.py --model LeNet --dataset mnist \
#      --epochs=20 \
#      --k_fold=1 \
#      --fullset=100 \
#      --stop_patience=20 \
#      --lambda_l2=0.0075632494949532695 \
#      --log=1 \
#      --strategy='{"optimizer": "sgd", "compression": "gradientsparsification", "learning_rate": 0.01, "k": 0.2,"max_iter": 2}'

