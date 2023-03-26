import tensorflow as tf
from tensorflow import keras
tf.config.run_functions_eagerly(True)


class AlexNet(keras.Model):
    def __init__(self, input_shape = (227, 227, 3), activ_func_Conv2D = 'relu', activ_func_dense = 'relu', activ_func_output = 'softmax', droprate = 0.5, num_classes = 8, **kwargs):
        super(AlexNet, self).__init__(**kwargs)
        # Block 1
        self.input_Conv2D = keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation=activ_func_Conv2D, input_shape=input_shape, padding="same", name="Block1_Conv")
        self.batchnorm1 = keras.layers.BatchNormalization()
        self.MaxPool2D1 = keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), name="Block1_pool")
        # Block 2
        self.Conv2D1 = keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation=activ_func_Conv2D, padding="same", name="Block2_Conv")
        self.batchnorm2 = keras.layers.BatchNormalization()
        self.MaxPool2D2 = keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), name="Block2_pool")
        # Block 3
        self.Conv2D2 = keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation=activ_func_Conv2D, padding="same", name="Block3_Conv1")
        self.batchnorm3 = keras.layers.BatchNormalization()
        self.Conv2D3 = keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation=activ_func_Conv2D, padding="same", name="Block3_Conv2")
        self.batchnorm4 = keras.layers.BatchNormalization()
        self.Conv2D4 = keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation=activ_func_Conv2D, padding="same", name="Block3_Conv3")
        self.batchnorm5 = keras.layers.BatchNormalization()
        self.MaxPool2D3 = keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), name="Block3_pool")
        # Classification block
        self.flatten_layer = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(4096, activation=activ_func_dense, name="fc1")
        self.dropout1 = keras.layers.Dropout(droprate)
        self.fc2 = keras.layers.Dense(4096, activation=activ_func_dense, name="fc2")
        self.dropout2 = keras.layers.Dropout(droprate)
        self.output_layer = keras.layers.Dense(num_classes, activation=activ_func_output, name="prediction")
        
    def call(self, inputs):
        X = self.input_Conv2D(inputs)
        X = self.batchnorm1(X)
        X = self.MaxPool2D1(X)
        X = self.Conv2D1(X)
        X = self.batchnorm2(X)
        X = self.MaxPool2D2(X)
        X = self.Conv2D2(X)
        X = self.batchnorm3(X)
        X = self.Conv2D3(X)
        X = self.batchnorm4(X)
        X = self.Conv2D4(X)
        X = self.batchnorm5(X)
        X = self.MaxPool2D3(X)
        X = self.flatten_layer(X)
        X = self.fc1(X)
        X = self.dropout1(X)
        X = self.fc2(X)
        X = self.dropout2(X)
        return self.output_layer(X)