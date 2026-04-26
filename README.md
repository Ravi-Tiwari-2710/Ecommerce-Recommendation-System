# 🛍️ E-commerce Product Recommendation System

A professional image-based product recommendation system that uses Deep Learning to find visually similar products.

## 🎯 Overview

This system implements a **Content-Based Filtering** approach. Instead of relying on user ratings, it analyzes the visual features of product images to suggest similar items. It leverages a pre-trained **ResNet50** model to extract high-dimensional feature vectors (embeddings) and uses **Cosine Similarity** to find the closest matches.

## 🛠️ Technical Architecture

- **Feature Extraction:** ResNet50 (Pre-trained on ImageNet).
- **Dimensionality Reduction:** GlobalMaxPooling2D to condense spatial information into a single feature vector.
- **Similarity Metric:** L2 Normalization + Cosine Similarity (Euclidean distance of normalized vectors).
- **Storage:** Python `pickle` for efficient storage of feature embeddings.

## 🚀 How It Works

1. **Image Processing:** All product images are resized to $224 \times 224$ pixels.
2. **Feature Mapping:** The image is passed through ResNet50 (without the top classification layer) to extract abstract visual patterns.
3. **Vectorization:** The output is flattened and normalized.
4. **Recommendation:** When a new image is provided, its embedding is compared against the pre-computed database of embeddings to find the most similar products.

## 🛠️ Installation & Usage

### Prerequisites
- Python 3.8+
- TensorFlow / Keras
- NumPy, Scikit-learn, Pillow

### Setup
1. **Clone the repo**
   ```bash
   git clone https://github.com/Ravi-Tiwari-2710/Ecommerce-Recommendation-System.git
   cd Ecommerce-Recommendation-System
   ```

2. **Prepare Images**
   Create a folder named `images/` in the root directory and add your product photos.

3. **Generate Embeddings**
   ```bash
   python main.py
   ```
   This will create `embeddings.pkl` and `filenames.pkl`.

---
**Developed by [Ravi Tiwari](https://github.com/Ravi-Tiwari-2710)**
