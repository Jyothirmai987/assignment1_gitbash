from training import CNN
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np

class test_model:
    def __init__(self):
        self.model = None

    def load_saved_model(self, model_path):
        # Load the saved model
        self.model = load_model(model_path)

    def test_img(self, image_path):
        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(28, 28), color_mode='grayscale')
        img_array = img_to_array(img)
        img_array /= 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict using the loaded model
        y_pred = self.model.predict(img_array)

        # Get the predicted label
        predicted_label = np.argmax(y_pred)

        return predicted_label
obj = test_model()
obj.load_saved_model("/content/mnist_cnn_model.h5")
predicted_label = obj.test_img('/content/drive/MyDrive/project_assignment/digit.jpg')
print("Predicted Label:", predicted_label)
