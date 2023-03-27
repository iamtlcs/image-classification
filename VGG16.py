import tensorflow as tf
from tensorflow import keras

class VGG16(keras.Model):
    def __init__(self, input_shape = (224, 224, 3), activ_Conv2D = 'relu', activ_dense = 'relu', activ_output = 'softmax', num_classes = 8, **kwargs):
        super(VGG16, self).__init__(**kwargs)
        # Block 1
        self.input_Conv2D = keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation=activ_Conv2D, input_shape=input_shape,  padding="same", name="Block1_Conv1")
        self.Conv2D1 = keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block1_Conv2")
        self.MaxPool2D1 = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), name="Block1_pool")
        # Block 2
        self.Conv2D2 = keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block2_Conv1")
        self.Conv2D3 = keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block2_Conv2")
        self.MaxPool2D2 = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), name="Block2_pool")
        # Block 3
        self.Conv2D4 = keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block3_Conv1")
        self.Conv2D5 = keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block3_Conv2")
        self.Conv2D6 = keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block3_Conv3")
        self.MaxPool2D3 = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), name="Block3_pool")
        # Block 4
        self.Conv2D7 = keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block4_Conv1")
        self.Conv2D8 = keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block4_Conv2")
        self.Conv2D9 = keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block4_Conv3")
        self.MaxPool2D4 = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), name="Block4_pool")
        # Block 5
        self.Conv2D10 = keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block5_Conv1")
        self.Conv2D11 = keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block5_Conv2")
        self.Conv2D12 = keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block5_Conv3")
        self.MaxPool2D5 = keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), name="Block5_pool")
        # Classification block
        self.flatten_layer = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(4096, activation=activ_dense, name="fc1")
        self.fc2 = keras.layers.Dense(4096, activation=activ_dense, name="fc2")
        self.output_layer = keras.layers.Dense(num_classes, activation=activ_output, name="prediction")
        
    def call(self, inputs):
        X = self.input_Conv2D(inputs)
        X = self.Conv2D1(X)
        X = self.MaxPool2D1(X)
        
        X = self.Conv2D2(X)
        X = self.Conv2D3(X)
        X = self.MaxPool2D2(X)
        
        X = self.Conv2D4(X)
        X = self.Conv2D5(X)
        X = self.Conv2D6(X)
        X = self.MaxPool2D3(X)
        
        X = self.Conv2D7(X)
        X = self.Conv2D8(X)
        X = self.Conv2D9(X)
        X = self.MaxPool2D4(X)
        
        X = self.Conv2D10(X)
        X = self.Conv2D11(X)
        X = self.Conv2D12(X)
        X = self.MaxPool2D5(X)
        
        X = self.flatten_layer(X)
        X = self.fc1(X)
        X = self.fc2(X)
        X = self.output_layer(X)
        return X