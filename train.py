# train.py
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf  # works the same for tf-nightly

# CONFIG
DATA_DIR = "data"   # <-- put your unzipped dataset here
IMAGE_SIZE = (224, 224)  # MobileNetV2 default
BATCH_SIZE = 32
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# 1) Discover classes
classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
print("Found classes:", classes)

# 2) Load images and labels
def load_images_and_labels(data_dir, classes, max_per_class=None):
    X_paths = []
    y = []
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if max_per_class:
            files = files[:max_per_class]
        for f in files:
            X_paths.append(f)
            y.append(cls)
    return X_paths, y

X_paths, y = load_images_and_labels(DATA_DIR, classes)
print("Total images:", len(X_paths))

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.joblib"))

# 3) Build MobileNetV2 feature extractor (no top)
base_model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), pooling='avg')
# pooling='avg' gives a fixed-length vector per image

def preprocess_image(path):
    img = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(img).astype("float32")
    # use the same preprocessing as Keras MobileNetV2
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return arr

# Extract features in batches to avoid memory blow up
def extract_features(paths, batch_size=32):
    features = []
    for i in tqdm(range(0, len(paths), batch_size), desc="Extracting features"):
        batch_paths = paths[i:i+batch_size]
        imgs = np.stack([preprocess_image(p) for p in batch_paths])
        feats = base_model.predict(imgs, verbose=0)
        features.append(feats)
    return np.vstack(features)

X_feats = extract_features(X_paths, batch_size=BATCH_SIZE)
print("Feature matrix shape:", X_feats.shape)

# 4) Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_feats)
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

# 5) Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# 6) Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("RandomForest classification report:")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))
joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.joblib"))

# 7) SVM (probabilities True)
# Warning: SVM with probability=True is slower to train; tune C/gamma if needed.
svc = SVC(probability=True, kernel='rbf', random_state=42)
# optional: grid search (comment out to save time)
# param_grid = {'C': [1, 10], 'gamma': ['scale', 0.01]}
# grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1, verbose=2)
# grid.fit(X_train, y_train)
# svc = grid.best_estimator_

svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
print("SVM classification report:")
print(classification_report(y_test, y_pred_svc, target_names=le.classes_))
joblib.dump(svc, os.path.join(MODELS_DIR, "svm.joblib"))

print("All models saved in", MODELS_DIR)
