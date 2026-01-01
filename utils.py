import os
import cv2
import numpy as np
from skimage.feature import hog

IMG_SIZE = 128


def load_images(folder, label):
    images = []
    labels = []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        img = cv2.imread(path)
        if img is None:
            continue

        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Noise removal (Gaussian Blur)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # Normalize
        img = img / 255.0

        images.append(img)
        labels.append(label)

    return images, labels


def extract_hog_features(images):
    features = []

    for img in images:
        feature = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )
        features.append(feature)

    return np.array(features)


def augment_image(image, rotation=0, flip=False, brightness=1.0):
    """
    Applies simple augmentation: rotation, flip, and brightness adjustment.
    """
    # Rotate
    if rotation != 0:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotation, 1.0)
        image = cv2.warpAffine(image, M, (w, h))

    # Flip
    if flip:
        image = cv2.flip(image, 1)  # Horizontal flip

    # Brightness (simple multiplication and clip)
    if brightness != 1.0:
        image = image.astype(np.float32)
        image = image * brightness
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image
