import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model = load_model('trained_model.keras')

# Load class indices
train_datagen = ImageDataGenerator(rescale=1./255)
train_flow = train_datagen.flow_from_directory(r'C:\Users\chavali sai sreekar\Desktop\MiniProject\Dataset\Train', target_size=(64, 64), batch_size=32, class_mode='categorical')
label_to_class = {v: k for k, v in train_flow.class_indices.items()}
print("Class indices:", train_flow.class_indices)
print("Label to class mapping:", label_to_class)

# Function to predict on a new image
def predict_image(image_path):
    new_image = tf.io.read_file(image_path)
    new_image = tf.image.decode_image(new_image, channels=3)
    new_image = tf.image.resize(new_image, (224, 224))  # Resize to 64x64 to match input size
    new_image = new_image / 255.0
    new_image = np.expand_dims(new_image, axis=0)  # Add batch dimension

    prediction = model.predict(new_image)
    print("Raw prediction output:", prediction)

    predicted_label = prediction.argmax()
    predicted_class = label_to_class[predicted_label]
    return predicted_class

# Example usage
new_image_path = r'C:\Users\chavali sai sreekar\Downloads\20240612_200432.jpg'
predicted_class = predict_image(new_image_path)
print(f'Prediction: {predicted_class}')
