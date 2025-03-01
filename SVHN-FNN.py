import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler

# Load SVHN Dataset
dataset, info = tfds.load("svhn_cropped", split=["train", "test"], as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset

def dataset_to_numpy(dataset):
    images, labels = [], []
    for img, label in tfds.as_numpy(dataset):
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)

X_train, y_train = dataset_to_numpy(train_dataset)
X_test, y_test = dataset_to_numpy(test_dataset)

# Normalize Data
mean = np.mean(X_train, axis=(0,1,2))
std = np.std(X_train, axis=(0,1,2))
X_train = (X_train - mean) / (std + 1e-7)
X_test = (X_test - mean) / (std + 1e-7)

# Flatten images
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# One-Hot Encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Split data into training and validation sets
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Learning Rate Scheduler
def lr_scheduler(epoch, lr):
    if epoch % 10 == 0 and epoch != 0:
        return lr * 0.1
    return lr

# Build Flexible Feedforward Neural Network
def build_model(hidden_layers=[128, 64], activation='relu', optimizer='adam', weight_decay=0):
    model = Sequential()
    model.add(Dense(hidden_layers[0], activation=activation, input_shape=(X_train.shape[1],),
                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation=activation,
                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
    
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Hyperparameters
hidden_layers = [128, 64, 32]
activation_function = 'relu'
optimizer_choice = tf.keras.optimizers.Adam(learning_rate=1e-3)
weight_decay_value = 0.0005
batch_size = 64
epochs = 10

# Train Model
model = build_model(hidden_layers=hidden_layers, activation=activation_function, optimizer=optimizer_choice, weight_decay=weight_decay_value)
lr_callback = LearningRateScheduler(lr_scheduler)
history = model.fit(
    X_train_split, y_train_split, 
    epochs=epochs,  
    batch_size=batch_size,  
    validation_data=(X_val, y_val), 
    callbacks=[lr_callback], 
    verbose=1
)

# Evaluate Model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f'Test Accuracy: {test_acc:.4f}')

# Plot Training History
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Model Loss')
plt.show()
