python model_train.py --model LeNet --dataset mnist \
      --epochs=20 \
      --k_fold=1 \
      --fullset=100 \
      --stop_patience=20 \
      --lambda_l2=0.0075632494949532695 \
      --log=1 \
      --strategy='{"optimizer": "sgd", "compression": "vqsgd", "learning_rate": 0.001, "repetition": 5}'

python model_train.py --model LeNet --dataset mnist \
      --epochs=20 \
      --k_fold=1 \
      --fullset=100 \
      --lambda_l2=0.0075632494949532695 \
      --stop_patience=20 \
      --log=1 \
      --strategy='{"optimizer": "sgd", "compression": "vqsgd", "learning_rate": 0.01, "repetition": 20}'

python model_train.py --model LeNet --dataset mnist \
      --epochs=20 \
      --k_fold=1 \
      --fullset=100 \
      --lambda_l2=0.0075632494949532695 \
      --stop_patience=20 \
      --log=1 \
      --strategy='{"optimizer": "sgd", "compression": "vqsgd", "learning_rate": 0.01, "repetition": 50}'

python model_train.py --model LeNet --dataset mnist \
      --epochs=20 \
      --k_fold=1 \
      --fullset=100 \
      --lambda_l2=0.0075632494949532695 \
      --stop_patience=20 \
      --log=1 \
      --strategy='{"optimizer": "sgd", "compression": "vqsgd", "learning_rate": 0.01, "repetition": 75}'

python model_train.py --model LeNet --dataset mnist \
      --epochs=20 \
      --k_fold=1 \
      --fullset=100 \
      --lambda_l2=0.0075632494949532695 \
      --stop_patience=20 \
      --log=1 \
      --strategy='{"optimizer": "sgd", "compression": "vqsgd", "learning_rate": 0.01, "repetition": 150}'