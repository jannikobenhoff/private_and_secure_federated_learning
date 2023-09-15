#!/bin/bash

echo "Starting the Gradient Sparsification script..."
/bin/bash /Users/jannikobenhoff/Documents/pythonProjects/private_and_secure_federated_learning/src/scripts/run_sgd.sh

if [ $? -eq 0 ]; then
    echo "Gradient Sparsification script completed successfully."
else
    echo "Gradient Sparsification script failed. Exiting..."
    exit 1
fi

echo "Starting the SGD script..."
/bin/bash /Users/jannikobenhoff/Documents/pythonProjects/private_and_secure_federated_learning/src/scripts/run_NaturalCompression.sh

if [ $? -eq 0 ]; then
    echo "SGD script completed successfully."
else
    echo "SGD script failed. Exiting..."
    exit 1
fi

echo "Starting the SGD script..."
/bin/bash /Users/jannikobenhoff/Documents/pythonProjects/private_and_secure_federated_learning/src/scripts/run_EFsignSGD.sh

if [ $? -eq 0 ]; then
    echo "SGD script completed successfully."
else
    echo "SGD script failed. Exiting..."
    exit 1
fi

echo "Starting the SGD script..."
/bin/bash /Users/jannikobenhoff/Documents/pythonProjects/private_and_secure_federated_learning/src/scripts/run_OneBitSGD.sh

if [ $? -eq 0 ]; then
    echo "SGD script completed successfully."
else
    echo "SGD script failed. Exiting..."
    exit 1
fi


echo "All scripts executed successfully."
