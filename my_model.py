from tensorflow import keras
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image

# Load models once when the app starts
model1 = keras.models.load_model('models/dataset1.h5')
model2 = keras.models.load_model('models/eye_disease_model.h5')
model3 = keras.models.load_model('models/model-3.h5')


def Integrated_Model(image_path):
    def model1_prediction(image_path):
        IMG_SIZE = 224

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image_data = cv2.imread(image_path)
        if image_data is None:
            raise ValueError(f"Unable to read image: {image_path}")

        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_data, (IMG_SIZE, IMG_SIZE))
        image_normalized = image_resized / 255.0
        image_input = np.expand_dims(image_normalized, axis=0)
        probabilities = model1.predict(image_input)
        class_names = ['other', 'ray']
        predicted_class = int(np.round(probabilities[0][0]))
        confidence = probabilities[0][0] if predicted_class == 1 else 1 - probabilities[0][0]
        return class_names[predicted_class], confidence * 100

    result, conf = model1_prediction(image_path)

    if result == 'ray':
        def model2_prediction(image_path):
            IMG_SIZE = 224

            image_data = cv2.imread(image_path)
            if image_data is None:
                raise ValueError(f"Unable to read image: {image_path}")

            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_data, (IMG_SIZE, IMG_SIZE))
            image_normalized = image_resized / 255.0
            image_input = np.expand_dims(image_normalized, axis=0)
            probabilities = model2.predict(image_input)
            class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
            return class_names[np.argmax(probabilities)], np.max(probabilities) * 100

        result, conf = model2_prediction(image_path)

        if result == 'diabetic_retinopathy':
            def model3_prediction(image_path):
                img = image.load_img(image_path, target_size=(128, 128))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                predicted_classes = model3.predict(img_array)
                predicted_class_index = np.argmax(predicted_classes)
                class_labels = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
                predicted_class_label = class_labels[predicted_class_index]
                if predicted_class_label == 'No_DR':
                    predicted_class_label = 'Mild'
                return predicted_class_label

            result = model3_prediction(image_path)
            return result, float(conf)

        else:
            return result, float(conf)

    else:
        return 'not ray', float(conf)
