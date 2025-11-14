import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tqdm import tqdm

# --------------------------
# Load MobileNetV2 Feature Extractor
# --------------------------
base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
feature_size = 1280

def extract_features(img_path):
    img = Image.open(img_path).resize((224, 224)).convert("RGB")
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    features = base_model.predict(arr)
    return features[0]   # shape: (1280,)

# --------------------------
# Dataset folders
# --------------------------
DATA_DIRS = [
    "dataset/train",
    "dataset/test"
]

images = []
labels = []

for DIR in DATA_DIRS:
    if not os.path.exists(DIR):
        print("❌ Folder not found:", DIR)
        continue

    for folder in os.listdir(DIR):
        class_path = os.path.join(DIR, folder)
        if os.path.isdir(class_path):
            print(f"📁 Loading class: {folder} from {DIR}")
            for file in tqdm(os.listdir(class_path), desc=f"Loading {folder}"):
                img_path = os.path.join(class_path, file)
                try:
                    feat = extract_features(img_path)
                    images.append(feat)
                    labels.append(folder)
                except:
                    pass

# Convert to NumPy
images = np.array(images)
labels = np.array(labels)

print("Total images loaded:", len(images))
print("Total labels loaded:", len(labels))

# --------------------------
# Label Encoding
# --------------------------
le = LabelEncoder()
labels_enc = le.fit_transform(labels)

# --------------------------
# Scaling features
# --------------------------
scaler = StandardScaler()
images_scaled = scaler.fit_transform(images)

# --------------------------
# Train SVM & RF
# --------------------------
svm = SVC(probability=True)
svm.fit(images_scaled, labels_enc)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(images_scaled, labels_enc)

# --------------------------
# Save models
# --------------------------
pickle.dump(le, open("label_encoder.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(svm, open("svm_model.pkl", "wb"))
pickle.dump(rf, open("rf_model.pkl", "wb"))
base_model.save("cnn_feature_model.keras")

print("🎉 Training complete! Models saved.")
