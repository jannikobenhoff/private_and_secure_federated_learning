## Installation

Using Conda (for Apple M1/M2):

1. ```conda env create -f tensorflow-apple-metal.yml -n tensorflow_fed```
2. ```conda activate tensorflow_fed```
3. ```python -m ipykernel install --user --name tensorflow_fed --display-name "Python 3.10 (tensorflow_fed)"```

## Run Compression Method

- run a script in ./scripts
- or ```cd src/scripts/``` followed by ```sh run_TernGrad.sh```

## Compression Methods

- Atomic Sparsification
- TernGrad
- EF-SignSGD
- FetchSGD
- Gradient Sparsification
- Sparsified SGD with Memory
- 1-Bit SGD
- Natural Compression
- Sparse Gradient
- TopK
- vqSGD