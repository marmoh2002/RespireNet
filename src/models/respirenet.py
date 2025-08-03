from tensorflow.keras.layers import (Input, Dense, Dropout, ReLU, GlobalAveragePooling2D,
                                     Concatenate, BatchNormalization)
from tensorflow.keras import Model
from .resnet import resnet34, resnet18


def respirenet(image_shape, tabular_shape, num_classes=2, resnet_body='34'):
    # --- Image Input Branch (ResNet backbone) ---
    image_input = Input(shape=image_shape, name='image_input')

    # Select ResNet backbone
    backbone = resnet34(
        image_shape) if resnet_body == '34' else resnet18(image_shape)
    x = backbone(image_input)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

    # --- Tabular Data Branch ---
    tabular_input = Input(shape=tabular_shape, name='tabular_input')
    y = Dense(32, activation='relu')(tabular_input)
    y = BatchNormalization()(y)
    y = Dropout(0.3)(y)

    # --- Feature Fusion ---
    combined = Concatenate()([x, y])

    # --- Combined Classifier ---
    z = Dense(128)(combined)
    z = ReLU()(z)
    z = Dropout(0.5)(z)

    z = Dense(128)(z)
    z = ReLU()(z)

    output = Dense(num_classes)(z)

    return Model(inputs=[image_input, tabular_input], outputs=output)
