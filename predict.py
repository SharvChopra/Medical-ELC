import cv2
import joblib
from skimage.feature import hog

IMG_SIZE = 128

# Load trained model
model = joblib.load("model.pkl")

def predict_mri(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = img / 255.0

    feature = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    result = model.predict([feature])[0]
    return "Tumor Detected" if result == 1 else "No Tumor"


# Example usage
print(predict_mri("test_mri.jpg"))
