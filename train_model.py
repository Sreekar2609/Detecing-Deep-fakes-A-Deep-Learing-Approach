import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Reshape, Permute, Multiply, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Directory paths
train_dir = r'C:\Users\chavali sai sreekar\Desktop\MiniProject\Dataset\Train'
validation_dir = r'C:\Users\chavali sai sreekar\Desktop\MiniProject\Dataset\Validation'
test_dir = r'C:\Users\chavali sai sreekar\Desktop\MiniProject\Dataset\Test'

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_flow = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
validation_flow = validation_datagen.flow_from_directory(validation_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
test_flow = validation_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

# Check the number of classes
num_classes = len(train_flow.class_indices)

# Reverse the class indices dictionary to map labels to class names
label_to_class = {v: k for k, v in train_flow.class_indices.items()}
print("Class indices:", train_flow.class_indices)
print("Label to class mapping:", label_to_class)

# Attention mechanism
def attention_block(inputs):
    # Global Average Pooling
    attention = GlobalAveragePooling2D()(inputs)
    attention = Dense(inputs.shape[-1], activation='sigmoid')(attention)
    attention = Reshape((1, 1, inputs.shape[-1]))(attention)
    attention = Multiply()([inputs, attention])
    return attention

# Model with attention mechanism
inputs = Input(shape=(224, 224, 3))

x = Conv2D(32, (3, 3), activation='relu')(inputs)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = attention_block(x)  # Add attention block

x = Flatten()(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_flow, epochs=10, validation_data=validation_flow)

# Evaluate the model
loss, accuracy = model.evaluate(test_flow)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Save the model
model.save('trained_model.keras')
