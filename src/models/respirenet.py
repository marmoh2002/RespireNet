from tensorflow.keras.layers import Conv2D, Dropout, Dense, ReLU, Input, Activation, Flatten, GlobalAveragePooling2D
from tensorflow.keras import Model

from .resnet import resnet34, resnet18


def respirenet(input_shape, num_classes=2, resnet_body='34'):
    inputs = Input(shape=input_shape, name='input')
    x = Conv2D(16, (2, 2), strides=(1, 1), padding='valid',
               kernel_initializer='normal')(inputs)
    x = AveragePooling2D((2, 2), strides=(1, 1))(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(16, (2, 2), strides=(1, 1), padding="valid",
               kernel_initializer='normal')(x)
    x = AveragePooling2D((2, 2), strides=(1, 1))(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, name='output_layer')(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=inputs, outputs=x)
    return Model(inputs=inputs, outputs=x)
