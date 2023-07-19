import os
from tensorboard import program

tracking_address = '../results/logs/scalars/baseline_model-20230718-181613'

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', os.path.abspath(tracking_address)])
    tb.main()
