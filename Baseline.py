import tensorflow as tf
from tensorflow import keras

class Baseline(keras.Model):
    def __init__(self, input_shape = (224, 224, 3), activ_Conv2D = 'relu', activ_dense = 'relu', activ_output = 'softmax', droprate = 0.5, num_classes = 8, **kwargs):
        super(Baseline, self).__init__(**kwargs)
        # Block 1
        self.norm1 = keras.layers.BatchNormalization(input_shape=input_shape, name="Block1_BatchNorm")
        self.Conv2D1 = keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block1_Conv")
        self.pool1 = keras.layers.MaxPool2D(pool_size=(3,3), name="Block1_Pool")
        self.dropout1 = keras.layers.Dropout(droprate/2, name="Block1_Dropout")
        #Block 2
        self.norm2 = keras.layers.BatchNormalization(input_shape=input_shape, name="Block2_BatchNorm")
        self.Conv2D2 = keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block2_Conv")
        self.pool2 = keras.layers.MaxPool2D(pool_size=(3,3), name="Block2_Pool")
        self.dropout2 = keras.layers.Dropout(droprate/2, name="Block2_Dropout")
        # Classification block
        self.flatten_layer = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(128, activation=activ_dense, name="fc")
        self.dropout3 = keras.layers.Dropout(droprate, name="Dropout")
        self.output_layer = keras.layers.Dense(num_classes, activation=activ_output, name="Prediction")
        
    def call(self, inputs):
        X = self.norm1(inputs)
        X = self.Conv2D1(X)
        X = self.pool1(X)
        X = self.dropout1(X)
        X = self.norm2(X)
        X = self.Conv2D2(X)
        X = self.pool2(X)
        X = self.dropout2(X)
        X = self.flatten_layer(X)
        X = self.fc1(X)
        X = self.dropout3(X)
        X = self.output_layer(X)
        return X