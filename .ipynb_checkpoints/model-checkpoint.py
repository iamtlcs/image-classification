import tensorflow as tf
from tensorflow import keras

def BaselineModel(input_shape = (224, 224, 3), activ_Conv2D = 'relu', activ_dense = 'relu', activ_output = 'softmax', droprate = 0.5, num_classes = 8):
    model = keras.models.Sequential([keras.layers.BatchNormalization(input_shape=input_shape, name="Block1_BatchNorm"),
                                     keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block1_Conv1"),
                                     keras.layers.MaxPool2D(pool_size=(3,3), name="Block1_Pool"),
                                     keras.layers.Dropout(droprate, name="Block1_Dropout"),
                                     keras.layers.BatchNormalization(name="Block2_BatchNorm"),
                                     keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block2_Conv1"),
                                     keras.layers.MaxPool2D(pool_size=(2,2), name="Block2_Pool"),
                                     keras.layers.Dropout(droprate, name="Block2_Dropout"),
                                     keras.layers.Flatten(name="Flatten_layer"),
                                     keras.layers.Dense(128, activation=activ_dense, name="fc"),
                                     keras.layers.Dropout(droprate, name="Dropout2"),
                                     keras.layers.Dense(num_classes, activation=activ_output, name="Prediction")
                                    ])
    return model

def AlexNetModel(input_shape = (224, 224, 3), activ_Conv2D = 'relu', activ_dense = 'relu', activ_output = 'softmax', droprate = 0.5, num_classes = 8):
    model = keras.models.Sequential([keras.layers.Conv2D(filters=32, kernel_size=(11,11), strides=(4,4), activation=activ_Conv2D, input_shape=input_shape, name="Block1_Conv"),
                                     keras.layers.BatchNormalization(name="Block1_BatchNorm"),
                                     keras.layers.MaxPool2D(pool_size=(2,2), name="Block1_Pool"),
                                     keras.layers.Conv2D(filters=64, kernel_size=(5,5), strides=(1,1), activation=activ_Conv2D, padding="same", name="Block2_Conv"),
                                     keras.layers.BatchNormalization(name="Block2_BatchNorm"),
                                     keras.layers.MaxPool2D(pool_size=(3,3), name="Block2_Pool"),
                                     keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation=activ_Conv2D, padding="same", name="Block3_Conv1"),
                                     keras.layers.BatchNormalization(name="Block3_BatchNorm1"),
                                     keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=(1,1), activation=activ_Conv2D, padding="same", name="Block3_Conv2"),
                                     keras.layers.BatchNormalization(name="Block3_BatchNorm2"),
                                     keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), activation=activ_Conv2D, padding="same", name="Block3_Conv3"),
                                     keras.layers.BatchNormalization(name="Block3_BatchNorm3"),
                                     keras.layers.MaxPool2D(pool_size=(2,2), name="Block3_Pool"),
                                     keras.layers.Flatten(name="Flatten_layer"),
                                     keras.layers.Dense(128, activation=activ_dense, name="fc1"),
                                     keras.layers.Dropout(droprate, name="Dropout1"),
                                     keras.layers.Dense(128, activation=activ_dense, name="fc2"),
                                     keras.layers.Dropout(droprate, name="Dropout2"),
                                     keras.layers.Dense(num_classes, activation=activ_output, name="Prediction")
                                    ])
    return model

def VGG16Model(input_shape = (224, 224, 3), activ_Conv2D = 'relu', activ_dense = 'relu', activ_output = 'softmax', num_classes = 8):
    model = keras.models.Sequential([keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation=activ_Conv2D, input_shape=input_shape,  padding="same", name="Block1_Conv1"),
                                     keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block1_Conv2"),
                                     keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), name="Block1_pool"),
                                     keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block2_Conv1"),
                                     keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block2_Conv2"),
                                     keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), name="Block2_pool"),
                                     keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block3_Conv1"),
                                     keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block3_Conv2"),
                                     keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block3_Conv3"),
                                     keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), name="Block3_pool"),
                                     keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block4_Conv1"),
                                     keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block4_Conv2"),
                                     keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block4_Conv3"),
                                     keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), name="Block4_pool"),
                                     keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block5_Conv1"),
                                     keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block5_Conv2"),
                                     keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block5_Conv3"),
                                     keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), name="Block5_pool"),
                                     keras.layers.Flatten(name="Flatten_layer"),
                                     keras.layers.Dense(4096, activation=activ_dense, name="fc1"),
                                     keras.layers.Dense(4096, activation=activ_dense, name="fc2"),
                                     keras.layers.Dense(num_classes, activation=activ_output, name="Prediction")
                                    ])
    return model

def VGG19Model(input_shape = (224, 224, 3), activ_Conv2D = 'relu', activ_dense = 'relu', activ_output = 'softmax', num_classes = 8):
    model = keras.models.Sequential([keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation=activ_Conv2D, input_shape=input_shape,  padding="same", name="Block1_Conv1"),
                                     keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block1_Conv2"),
                                     keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), name="Block1_pool"),
                                     keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block2_Conv1"),
                                     keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block2_Conv2"),
                                     keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), name="Block2_pool"),
                                     keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block3_Conv1"),
                                     keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block3_Conv2"),
                                     keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block3_Conv3"),
                                     keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block3_Conv4"),
                                     keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), name="Block3_pool"),
                                     keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block4_Conv1"),
                                     keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block4_Conv2"),
                                     keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block4_Conv3"),
                                     keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block4_Conv4"),
                                     keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), name="Block4_pool"),
                                     keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block5_Conv1"),
                                     keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block5_Conv2"),
                                     keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block5_Conv3"),
                                     keras.layers.Conv2D(filters=512, kernel_size=(3,3), activation=activ_Conv2D, padding="same", name="Block5_Conv4"),
                                     keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), name="Block5_pool"),
                                     keras.layers.Flatten(name="Flatten_layer"),
                                     keras.layers.Dense(4096, activation=activ_dense, name="fc1"),
                                     keras.layers.Dense(4096, activation=activ_dense, name="fc2"),
                                     keras.layers.Dense(num_classes, activation=activ_output, name="Prediction")
                                    ])
    return model