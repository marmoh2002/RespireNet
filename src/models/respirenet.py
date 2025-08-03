from tensorflow.keras.layers import Input, Dense, Dropout, ReLU, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.models import Model

# Make sure your resnet import is correct based on your file structure
# Assuming it's in the same directory for this example.
from .resnet import resnet34, resnet18


def respirenet_multi_input(
    image_shape,
    tabular_shape,
    num_classes=2,
    resnet_body='34'
):
    # --- 1. Define Input Layers (This was already correct) ---
    image_input = Input(shape=image_shape, name='image_input')
    tabular_input = Input(shape=tabular_shape, name='tabular_input')

    # --- 2. Build the CNN Branch (Updated Logic) ---
    # Instantiate the ResNet backbone as a layer
    if resnet_body == '34':
        backbone = resnet34()  # Returns the ResNet layer
    else:
        backbone = resnet18()  # Returns the ResNet layer

    # Call the backbone layer on your specific input tensor.
    # This correctly connects your `image_input` to the ResNet layers.
    cnn_branch = backbone(image_input)

    # The rest of the CNN branch
    cnn_branch = GlobalAveragePooling2D()(cnn_branch)

    # --- 3. Build the MLP Branch (No changes needed) ---
    mlp_branch = Dense(32, activation='relu')(tabular_input)
    mlp_branch = Dropout(0.3)(mlp_branch)
    mlp_branch = Dense(16, activation='relu')(mlp_branch)

    # --- 4. Concatenate (No changes needed) ---
    combined_features = Concatenate()([cnn_branch, mlp_branch])

    # --- 5. Classifier Head (No changes needed) ---
    x = Dense(128, activation='relu')(combined_features)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # --- 6. Create the Model (No changes needed) ---
    # The inputs are now correctly wired to the rest of the model.
    model = Model(inputs=[image_input, tabular_input], outputs=outputs)

    return model

# --- How to Use It ---

# # Define the shapes of your inputs
# # This must match the data coming from your CoswaraCovidDataset
# IMAGE_SHAPE = (128, 431, 1)  # Example: (Mels, Time_Steps, Channels)
# TABULAR_SHAPE = (4,)         # Example: (age_normalized + 3 one-hot sex features)
# NUM_CLASSES = 2

# # Create the model instance
# model = respirenet_multi_input(
#     image_shape=IMAGE_SHAPE,
#     tabular_shape=TABULAR_SHAPE,
#     num_classes=NUM_CLASSES,
#     resnet_body='34'
# )

# # Compile the model
# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy', # Use this for one-hot encoded labels
#     metrics=['accuracy']
# )

# # Print the model summary to verify the architecture
# model.summary()
