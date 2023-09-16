#!/bin/bash

echo "Starting the SGD script..."
/bin/bash /Users/jannikobenhoff/Documents/pythonProjects/private_and_secure_federated_learning/src/scripts/run_NaturalCompression.sh

if [ $? -eq 0 ]; then
    echo "SGD script completed successfully."
else
    echo "SGD script failed. Exiting..."
    exit 1
fi

echo "Starting the SGD script..."
/bin/bash /Users/jannikobenhoff/Documents/pythonProjects/private_and_secure_federated_learning/src/scripts/run_TernGrad.sh

if [ $? -eq 0 ]; then
    echo "SGD script completed successfully."
else
    echo "SGD script failed. Exiting..."
    exit 1
fi


echo "All scripts executed successfully."
