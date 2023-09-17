names = {
    "sgd terngrad": "TernGrad",
    "terngrad": "TernGrad",
    "sgd naturalcompression": "Natural Compression",
    "naturalcompression": "Natural Compression",
    "sgd onebitsgd": "1-Bit SGD",
    "onebitsgd": "1-Bit SGD",
    "sgd sparsegradient": "Sparse Gradient",
    "sparsegradient": "Sparse Gradient",
    "sgd gradientsparsification": "Gradient Sparsification",
    "gradientsparsification": "Gradient Sparsification",
    "memsgd": "Sparsified SGD with Memory",
    "memsgd none": "Sparsified SGD with Memory",
    "memsgd ": "Sparsified SGD with Memory",
    "atomo": "Atomic Sparsification",
    "sgd atomo": "Atomic Sparsification",
    "efsignsgd": "EF-SignSGD",
    "efsignsgd ": "EF-SignSGD",
    "fetchsgd": "FetchSGD",
    "fetchsgd ": "FetchSGD",
    "vqsgd": "vqSGD",
    "sgd vqsgd": "vqSGD",
    "topk": "Top-K",
    "sgd topk": "Top-K",
    "sgd ": "SGD",
    "sgd": "SGD",
    "sgdm": "SGD with Momentum",
    "sgdm ": "SGD with Momentum",
    "sgd_vgg": "SGD",
    "sgd none": "SGD",
    "sgd bsgd": "BucketSGD",
    "bsgd": "BucketSGD",
}

colors = {
    "SGD": 'b', 'SGD with Momentum': "darkgreen",
    "BucketSGD": 'g', "vqSGD": "#32CD32", "Top-K": 'y',
    "EF-SignSGD": 'm', "FetchSGD": 'c', "Atomic Sparsification": 'grey', "Sparsified SGD with Memory": 'orange',
    "Natural Compression": 'pink', "Gradient Sparsification": "#D2691E", "1-Bit SGD": 'r', "TernGrad": "#FFD700",
    "Sparse Gradient": "#a6bddb"
}

markers = {
    "SGD": 'o',  # circle
    'SGD with Momentum': ".",
    "BucketSGD": 's',  # square
    "vqSGD": '^',  # triangle up
    "Top-K": 'D',  # diamond
    "EF-SignSGD": 'p',  # pentagon
    "FetchSGD": '*',  # star
    "Atomic Sparsification": 'H',  # hexagon1
    "Sparsified SGD with Memory": 'h',  # hexagon2
    "Natural Compression": '8',  # octagon
    "Gradient Sparsification": 'P',  # plus (filled)
    "1-Bit SGD": 'X',  # x (filled)
    "TernGrad": 'v',  # triangle down
    "Sparse Gradient": "s"
}
