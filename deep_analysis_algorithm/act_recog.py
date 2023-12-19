import cv2
import numpy as np
from tensorflow.keras.models import load_model
# from google.colab import drive

# drive.mount('/content/drive/')


IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 4
CLASSES_LIST = ["smoke", "shoot_gun", "walk", "hit"]


convlstm_model_h5 = load_model(
    'D:/pycharm/Ai Cam/simple_deep_sort/convlstm_model.h5', compile=False)
convlstm_model_h5.compile(
    optimizer="rmsprop",
    loss=None,
    metrics=None,
    loss_weights=None,
    weighted_metrics=None,
    run_eagerly=None,
    steps_per_execution=None,
    jit_compile=None
)

# lib_version

def predict_single_action(frames_list):
    predicted_class_name = ''

    predicted_labels_probabilities = convlstm_model_h5.predict(np.expand_dims(frames_list, axis = 0))[0]

    predicted_label = np.argmax(predicted_labels_probabilities)

    predicted_class_name = CLASSES_LIST[predicted_label]

    print(f'Action Predicted: {predicted_class_name}/nConfidence: {predicted_labels_probabilities[predicted_label]}')

    return [predicted_class_name,predicted_labels_probabilities[predicted_label]]

predict_single_action()