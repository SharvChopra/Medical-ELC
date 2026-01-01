import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import time
from skimage.feature import hog
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import load_images, extract_hog_features, augment_image

# Set Page Config
st.set_page_config(page_title="Brain Tumor Detection - Assessment Modules", layout="wide")

st.title("Brain Tumor Detection System")
st.markdown("### Assessment Modules 1-5")

# Sidebar for Navigation
module_selection = st.sidebar.radio(
    "Select Assessment Module",
    [
        "Module 1: Image Data Analysis",
        "Module 2: Preprocessing",
        "Module 3: Feature Extraction & Modeling",
        "Module 4: Evaluation Metrics",
        "Module 5: Real-time Deployment"
    ]
)

# Constants
DATA_DIR = "brain_tumor_dataset"
IMG_SIZE = 128

@st.cache_resource
def load_raw_data():
    """
    Loads raw file paths and counts for analysis.
    """
    yes_path = os.path.join(DATA_DIR, "yes")
    no_path = os.path.join(DATA_DIR, "no")
    
    yes_files = [os.path.join(yes_path, f) for f in os.listdir(yes_path)]
    no_files = [os.path.join(no_path, f) for f in os.listdir(no_path)]
    
    return yes_files, no_files

@st.cache_resource
def get_cached_dataset():
    """
    Loads processed dataset from utils.
    """
    tumor_imgs, tumor_labels = load_images(os.path.join(DATA_DIR, "yes"), 1)
    normal_imgs, normal_labels = load_images(os.path.join(DATA_DIR, "no"), 0)
    
    X = np.array(tumor_imgs + normal_imgs)
    y = np.array(tumor_labels + normal_labels)
    return X, y

# --- MODULE 1: Image Data Analysis ---
if module_selection == "Module 1: Image Data Analysis":
    st.header("Module 1: Image Data Analysis (EDA)")
    
    yes_files, no_files = load_raw_data()
    
    st.subheader("1. Dataset Stats")
    col1, col2 = st.columns(2)
    col1.metric("Tumor Images (Yes)", len(yes_files))
    col2.metric("Normal Images (No)", len(no_files))
    
    st.subheader("2. Sample Visualizations")
    if st.button("Show New Random Samples"):
        img_path_yes = np.random.choice(yes_files)
        img_path_no = np.random.choice(no_files)
        
        c1, c2 = st.columns(2)
        with c1:
            st.image(img_path_yes, caption="Brain Tumor (Yes)", use_column_width=True)
            st.text(f"Dimensions: {cv2.imread(img_path_yes).shape}")
        with c2:
            st.image(img_path_no, caption="Healthy (No)", use_column_width=True)
            st.text(f"Dimensions: {cv2.imread(img_path_no).shape}")

    st.subheader("3. Pixel Intensity Distribution")
    if len(yes_files) > 0:
        sample_img = cv2.imread(yes_files[0], cv2.IMREAD_GRAYSCALE)
        fig, ax = plt.subplots()
        ax.hist(sample_img.ravel(), bins=256, range=[0, 256])
        ax.set_title("Pixel Intensity Histogram (Sample)")
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        
    st.info("The dataset contains MRI scans with varying dimensions and intensity distributions. Some images may have noise or artifacts.")

# --- MODULE 2: Preprocessing ---
elif module_selection == "Module 2: Preprocessing":
    st.header("Module 2: Preprocessing Image Data")
    st.markdown("Perform resizing, grayscale conversion, noise removal, and augmentation.")
    
    yes_files, _ = load_raw_data()
    sample_file = st.selectbox("Select a sample image for preprocessing", yes_files[:10])
    
    # Load original
    original_img = cv2.imread(sample_file)
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Sidebar controls for Module 2
    st.write("### Apply Filters")
    do_resize = st.checkbox("Resize (128x128)", value=True)
    do_gray = st.checkbox("Convert to Grayscale", value=True)
    blur_kernel = st.slider("Gaussian Blur Kernel Size (Odds only)", 1, 15, 5, step=2)
    
    st.write("### Augmentation")
    rotation = st.slider("Rotation (degrees)", -180, 180, 0)
    flip = st.checkbox("Horizontal Flip", value=False)
    brightness = st.slider("Brightness Factor", 0.5, 2.0, 1.0)
    
    # Process
    processed = original_img.copy()
    
    # 1. Resize
    if do_resize:
        processed = cv2.resize(processed, (IMG_SIZE, IMG_SIZE))
    
    # 2. Augment (Apply on color or gray?) - Applying on current state
    processed = augment_image(processed, rotation=rotation, flip=flip, brightness=brightness)

    # 3. Grayscale
    if do_gray:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
    # 4. Blur
    if blur_kernel > 1:
        processed = cv2.GaussianBlur(processed, (blur_kernel, blur_kernel), 0)
        
    # Visualization
    c1, c2 = st.columns(2)
    with c1:
        st.image(original_rgb, caption="Original Image", use_column_width=True)
    with c2:
        st.image(processed, caption="Preprocessed Image", use_column_width=True, channels="GRAY" if do_gray else "RGB")

# --- MODULE 3: Feature Extraction & Modeling ---
elif module_selection == "Module 3: Feature Extraction & Modeling":
    st.header("Module 3: Feature Extraction & ML Model")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Feature Extraction (HOG)")
        st.write("Histogram of Oriented Gradients (HOG) captures edge directions.")
        
        yes_files, _ = load_raw_data()
        sample_file = yes_files[0]
        img = cv2.imread(sample_file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True)

        st.image(hog_image, caption="HOG Feature Visualization", use_column_width=True, clamp=True)
        st.write(f"Feature Vector Length: {len(fd)}")
        
    with col2:
        st.subheader("2. Train Model (SVM)")
        if st.button("Train SVM Model"):
            with st.spinner("Loading data and extracting features..."):
                X, y = get_cached_dataset()
                # Features are already extracted? No, load_images does preprocessing, but not HOG.
                # utils.load_images returns images. We need to extract features.
                X_features = extract_hog_features(X)
                
                # Split
                X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
                
                # Model
                model = SVC(kernel="linear", probability=True)
                model.fit(X_train, y_train)
                
                # Save to session state
                st.session_state['model'] = model
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['X_train'] = X_train
                st.session_state['y_train'] = y_train
                
                acc = model.score(X_test, y_test)
                st.success(f"Model Trained! Accuracy: {acc:.2%}")
                
        # Description
        st.info("""
        **Short Note**:
        - **Features**: HOG (Histogram of Oriented Gradients) extracts shape and edge information.
        - **Why**: Medical images rely heavily on structural anomalies (tumors) which HOG captures well.
        - **Model**: Linear SVM is effective for high-dimensional HOG features.
        """)

# --- MODULE 4: Evaluation ---
elif module_selection == "Module 4: Evaluation Metrics":
    st.header("Module 4: Model Evaluation")
    
    if 'model' not in st.session_state:
        st.warning("Please train the model in Module 3 first.")
    else:
        model = st.session_state['model']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        X_train = st.session_state['X_train']
        y_train = st.session_state['y_train']
        
        # 1. Metrics
        st.subheader("1. Classification Metrics")
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        st.metric("Test Accuracy", f"{acc:.2%}")
        
        st.text("Classification Report:")
        st.code(classification_report(y_test, y_pred))
        
        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
               xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
               ylabel='True label', xlabel='Predicted label')
        
        # Loop over data dimensions and create text annotations.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2. else "black")
        st.pyplot(fig)
        
        # 2. Learning Curves (Simulated)
        st.subheader("2. Training vs Validation Performance (Learning Curve)")
        with st.spinner("Generating Learning Curve..."):
            train_sizes, train_scores, test_scores = learning_curve(
                model, np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test)), 
                cv=3, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
            )
            
            train_scores_mean = np.mean(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            
            fig_lc, ax_lc = plt.subplots()
            ax_lc.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
            ax_lc.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
            ax_lc.set_title("Learning Curve")
            ax_lc.set_xlabel("Training examples")
            ax_lc.set_ylabel("Score")
            ax_lc.legend(loc="best")
            ax_lc.grid()
            st.pyplot(fig_lc)
        
        st.write("The gap between training and validation scores indicates how well the model generalizes.")

# --- MODULE 5: Deployment ---
elif module_selection == "Module 5: Real-time Deployment":
    st.header("Module 5: Real-time Deployment Simulation")
    
    st.write("### 1. Upload MRI Scan")
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])
    
    if 'model' not in st.session_state:
        st.warning("Please train the model in Module 3 first!")
    else:
        model = st.session_state['model']
        
        if uploaded_file is not None:
            # Read
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            
            # Display
            c1, c2 = st.columns(2)
            c1.image(img, caption="Uploaded Image", use_column_width=True)
            
            if c1.button("Predict"):
                start_time = time.time()
                
                # Preprocess same as training
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
                img_blur = cv2.GaussianBlur(img_resized, (5, 5), 0)
                img_norm = img_blur / 255.0
                
                # Feature Extraction
                feat = hog(img_norm, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), block_norm='L2-Hys')
                
                # Predict
                prediction = model.predict([feat])[0]
                proba = model.predict_proba([feat])[0]
                
                end_time = time.time()
                latency = end_time - start_time
                
                label = "Tumor Detected" if prediction == 1 else "No Tumor"
                confidence = proba[prediction]
                
                with c2:
                    st.metric("Result", label)
                    st.metric("Confidence", f"{confidence:.2%}")
                    st.metric("Latency", f"{latency:.4f} sec")
                    
                    if prediction == 1:
                        st.error("Anomaly Detected!")
                    else:
                        st.success("Normal Scan")
                        
    st.write("---")
    st.subheader("4. Deployment Analysis")
    st.write("""
    - **Response Time**: The system responds in < 1 second for single image predictions.
    - **Stability**: Predictions are consistent with the trained SVM model.
    - **Challenges**: Ensuring the uploaded image format and quality matches the training data (preprocessing consistency).
    """)
