import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
print(tf.__version__)

#### Import the Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# Load model
model = tf.keras.models.load_model('mnist_model.h5')

# Create GUI
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        # Create canvas to display image
        self.canvas = tk.Canvas(self, width=300, height=300)
        self.canvas.pack()

        # Create button to open file dialog
        self.button = tk.Button(self, text='Open', command=self.load_image)
        self.button.pack()

        # Create button to classify image
        self.classify_button = tk.Button(self, text='Classify', command=self.classify_image)
        self.classify_button.pack()

        # Create label to display classification result
        self.label = tk.Label(self, text='')
        self.label.pack()

    def load_image(self):
        # Open file dialog to choose image
        file_path = filedialog.askopenfilename()

        # Load image and resize to 28x28 pixels
        image = Image.open(file_path).convert('L')
        image = image.resize((28, 28))

        # Display image on canvas
        self.image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor='nw', image=self.image)

        # Convert image to numpy array
        self.image_array = np.array(image)

    def classify_image(self):
        # Reshape image array to match input shape of model
        input_array = self.image_array.reshape(1, 28, 28, 1) / 255.0

        # Use model to predict digit
        prediction = model.predict(input_array)

        # Get digit with highest probability
        digit = np.argmax(prediction)

        # Display result on label
        self.label.config(text=f'Predicted digit: {digit}')

# Create and run application
root = tk.Tk()
app = Application(master=root)
app.mainloop()