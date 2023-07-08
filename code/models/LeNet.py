from tensorflow.keras import models, layers

class LeNet:
    def __init__(self):
        self.model = models.Sequential()
        self.model.add