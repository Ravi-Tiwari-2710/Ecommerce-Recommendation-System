import os
import numpy as np
import pickle
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm

def create_feature_extractor():
    """
    Builds a feature extraction model using ResNet50.
    """
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])
    return model

def extract_features(img_path, model):
    """
    Extracts normalized feature vectors from an image.
    """
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def main():
    # --- Configuration ---
    # Replace with your actual images directory path
    IMAGE_DIR = 'images' 
    EMBEDDINGS_PATH = 'embeddings.pkl'
    FILENAMES_PATH = 'filenames.pkl'
    
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Directory {IMAGE_DIR} not found. Please create it and add images.")
        return

    print("Initializing Feature Extractor (ResNet50)...")
    model = create_feature_extractor()

    filenames = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not filenames:
        print("No valid images found in the directory.")
        return

    print(f"Extracting features from {len(filenames)} images...")
    feature_list = []
    valid_filenames = []

    for file in tqdm(filenames):
        features = extract_features(file, model)
        if features is not None:
            feature_list.append(features)
            valid_filenames.append(file)

    # Save results
    print(f"Saving embeddings for {len(valid_filenames)} images...")
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(feature_list, f)
    with open(FILENAMES_PATH, 'wb') as f:
        pickle.dump(valid_filenames, f)

    print("✅ Process completed successfully!")

if __name__ == '__main__':
    main()
