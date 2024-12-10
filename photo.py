import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os
import cv2
import marshal
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Charger le modèle VGG16 pré-entraîné
model = VGG16(weights='imagenet', include_top=False, pooling='avg')
features_list = np.loadtxt("features_list.txt", dtype=int)
image_paths = marshal.load(open("image_paths", "rb"))

def extract_features(img_array):
    img_data = np.expand_dims(img_array, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()



def search_image(query_image_array, features_list, image_paths, k=3):
    query_features = extract_features(query_image_array)
    knn = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean').fit(features_list)
    distances, indices = knn.kneighbors([query_features])
    return [image_paths[i] for i in indices[0]]

class WebcamApp:
    def __init__(self, root, features_list, image_paths):
        self.root = root
        self.features_list = features_list
        self.image_paths = image_paths
        self.cap = cv2.VideoCapture(0)
        self.is_capturing = False

        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        self.capture_button = ttk.Button(root, text="Prendre une photo", command=self.capture_image)
        self.capture_button.pack()

        self.update()

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.root.after(10, self.update)

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            query_image_array = np.array(frame)
            similar_images = search_image(query_image_array, self.features_list, self.image_paths)

            print("Images similaires trouvées :")
            for img_path in similar_images:
                print(img_path)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

root = tk.Tk()
app = WebcamApp(root, features_list, image_paths)
root.mainloop()
