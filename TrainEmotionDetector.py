# Import necessary libraries
import cv2  # OpenCV for computer vision tasks
from keras.models import Sequential  # To build a sequential neural network
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten  # Layers for CNN
from tensorflow.keras.optimizers import Adam  # Optimizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Data augmentation and preprocessing

# Initialize image data generator for training and validation datasets with rescaling
training_data_generator = ImageDataGenerator(rescale=1./255)  # Normalize pixel values to [0, 1]
validation_data_generator = ImageDataGenerator(rescale=1./255)  # Normalize validation data

# Preprocess all training images
train_generator = training_data_generator.flow_from_directory(
        'data/train',  # Directory path for training images
        target_size=(48, 48),  # Resize images to 48x48 pixels
        batch_size=64,  # Number of samples per batch
        color_mode="grayscale",  # Use grayscale images
        class_mode='categorical')  # One-hot encode class labels

# Preprocess all validation images
validation_generator = validation_data_generator.flow_from_directory(
        'data/test',  # Directory path for validation images
        target_size=(48, 48),  # Resize images to 48x48 pixels
        batch_size=64,  # Number of samples per batch
        color_mode="grayscale",  # Use grayscale images
        class_mode='categorical')  # One-hot encode class labels

# Create a CNN model structure for emotion recognition
machine_learning_model = Sequential()  # Initialize a sequential model

# Add the first convolutional layer
machine_learning_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))

# Add a second convolutional layer
machine_learning_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

# Add a max pooling layer to reduce dimensionality
machine_learning_model.add(MaxPooling2D(pool_size=(2, 2)))

# Add dropout to reduce overfitting
machine_learning_model.add(Dropout(0.25))

# Add third convolutional layer
machine_learning_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

# Add a second max pooling layer
machine_learning_model.add(MaxPooling2D(pool_size=(2, 2)))

# Add fourth convolutional layer
machine_learning_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

# Add a third max pooling layer
machine_learning_model.add(MaxPooling2D(pool_size=(2, 2)))

# Add another dropout layer to further prevent overfitting
machine_learning_model.add(Dropout(0.25))

# Flatten the 2D output to 1D for fully connected layers
machine_learning_model.add(Flatten())

# Add a dense (fully connected) layer with ReLU activation
machine_learning_model.add(Dense(1024, activation='relu'))

# Add a dropout layer for regularization
machine_learning_model.add(Dropout(0.5))

# Add the output layer with 7 nodes (one for each emotion) and softmax activation
machine_learning_model.add(Dense(7, activation='softmax'))

# Disable OpenCL acceleration in OpenCV to avoid conflicts
cv2.ocl.setUseOpenCL(False)

# Compile the model with categorical cross-entropy loss, Adam optimizer, and accuracy metric
machine_learning_model.compile(
    loss='categorical_crossentropy',  # Suitable for multi-class classification
    optimizer=Adam(learning_rate=0.0001, decay=1e-6),  # Learning rate and decay
    metrics=['accuracy']  # Evaluate model performance using accuracy
)

# Train the CNN model
model_info = machine_learning_model.fit(
        train_generator,  # Training data generator
        steps_per_epoch=28709 // 64,  # Total number of batches per epoch
        epochs=50,  # Number of epochs to train
        validation_data=validation_generator,  # Validation data generator
        validation_steps=7178 // 64)  # Total number of validation batches

# Save the model architecture to a JSON file
model_json = machine_learning_model.to_json()
with open("machine_learning_model.json", "w") as json_file:
    json_file.write(model_json)

# Save the trained model weights to an H5 file
machine_learning_model.save_weights('machine_learning_model.weights.h5')

