# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:00:06 2024

@author: PIXEL
"""

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
import json

# 加载配置文件
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# Load the trained model
model = load_model(config['model']['model_file'])

# Define the image dimensions 
img_height = config['image_processing']['target_size'][0]
img_width = config['image_processing']['target_size'][1]

 # Replace class names
class_labels = config['ui']['class_labels'] 

# Create the main window
root = tk.Tk()
root.title(config['ui']['window_title'])

# Set window size
root.geometry(config['ui']['window_size'])

image_label = tk.Label(root)
image_label.pack(pady=20)

result_label = tk.Label(root, text="", font=tuple(config['ui']['font']))
result_label.pack()

def preprocess_image(image_path):
    """Preprocess the image to the required size and shape."""
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    # Rescale as done during training
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def classify_image():
    """Classify the selected image and display the result."""
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        try:
            # Preprocess the image
            processed_image = preprocess_image(file_path)
            
            prediction = model.predict(processed_image)
            predicted_class = np.where(prediction > config['model']['test_prediction_threshold'], 1, 0)
            
            predicted_label = class_labels[predicted_class[0][0]]
            
            img = Image.open(file_path)
            img = img.resize(tuple(config['ui']['image_display_size']))
            img_tk = ImageTk.PhotoImage(img)
            image_label.configure(image=img_tk)
            image_label.image = img_tk 
            
            # Display the prediction
            result_label.config(text=f"Predicted Class: {predicted_label}")
        except Exception as e:
            result_label.config(text=f"Error: {str(e)}")

classify_button = tk.Button(root, text="Select Image", command=classify_image)
classify_button.pack(pady=10)

root.mainloop()
