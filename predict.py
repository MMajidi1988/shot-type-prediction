import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt

def load_and_predict_images(folder_path, model_path):
    # Load the pre-trained model
    model = load_model(model_path)
    print("Model loaded successfully.")

    # Specify the image size (must match the model input size)
    image_size = (224, 224)  # Adjust based on your model's expected input

    # Mapping of class indices to class names
    class_names = {0: '[close-up]', 1: '[medium]', 2: '[full]'}

    # Loop through each image in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):  # Assuming the images are in jpg format
            img_path = os.path.join(folder_path, filename)
            image = load_img(img_path, target_size=image_size)
            image_array = img_to_array(image)
            image_preprocessed = preprocess_input(image_array)
            image_preprocessed = np.expand_dims(image_preprocessed, axis=0)

            # Predict the class of the image
            prediction = model.predict(image_preprocessed)
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_class_name = class_names[predicted_class_index]
            
            # Print the filename and the predicted class
            print(f"Filename: {filename}, Predicted class: {predicted_class_name}")

            # Display the image with the prediction in the title of the popup window
            plt.figure(figsize=(8, 8))  # Optional: Adjust the figure size as needed
            plt.imshow(image_array.astype('uint8'))  # Convert back to uint8 to display correctly
            plt.title(f"Filename: {filename}\nPredicted class: {predicted_class_name}")
            plt.axis('off')  # Turn off axis numbers and ticks
            plt.show()




if __name__ == "__main__":
    # Path to the folder containing images
    images_folder = "/home/martin/Hockey/last-hockey-version/others/shottypes/images-model-test"
    
    # Path to the saved model
    model_path = "/home/martin/Hockey/last-hockey-version/others/shottypes/best_model.keras"
    
    # Run the prediction function
    load_and_predict_images(images_folder, model_path)
