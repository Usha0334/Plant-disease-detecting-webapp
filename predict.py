from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

def preprocess_image(img):
    img = img.resize((224, 224)).convert("RGB")
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

def predict(img, le, scaler, rf, svm, base_model):

    arr = preprocess_image(img)

    # CNN feature extraction → 1280-dim vector
    features = base_model.predict(arr)          # shape (1, 1280)
    features_scaled = scaler.transform(features) # scale

    # Get probabilities
    probs_rf = rf.predict_proba(features_scaled)[0]
    probs_svm = svm.predict_proba(features_scaled)[0]

    # Average both models
    probs = (probs_rf + probs_svm) / 2

    top_idx = np.argsort(probs)[::-1]

    return probs, top_idx
