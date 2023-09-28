import os
from tensorboard import program

tracking_address = '../federated/logs/20230904-160759'

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', os.path.abspath(tracking_address)])
    tb.main()
