# predict.py
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Model path (uses the path you provided)
MODEL_PATH = Path(__file__).parent / "best_model.h5"

# Load once on import
model = load_model(str(MODEL_PATH))

# Exact class names your model was trained on
CLASS_NAMES = ["Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy"]

def preprocess_image(pil_image: Image.Image, target_size=(224, 224)):
    """Return a (1, H, W, 3) float32 numpy array in [0,1]."""
    if not isinstance(pil_image, Image.Image):
        pil_image = Image.fromarray(pil_image)
    img = pil_image.convert("RGB").resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def predict(pil_image: Image.Image):
    """
    Returns:
      predicted_class (str),
      confidence (float),
      preds (np.array) -- raw softmax vector,
      img_array (np.array) -- preprocessed (1,H,W,3)
    """
    img_array = preprocess_image(pil_image)
    preds = model.predict(img_array)
    preds = np.asarray(preds)  # ensure numpy
    preds0 = preds[0]
    class_idx = int(np.argmax(preds0))
    predicted_class = CLASS_NAMES[class_idx]
    confidence = float(preds0[class_idx])
    return predicted_class, confidence, preds0, img_array
