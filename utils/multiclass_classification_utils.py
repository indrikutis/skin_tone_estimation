import os
import sys
import numpy as np
import dlib
import PIL.Image
import cv2
import scipy.ndimage
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import json
import random
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
from sklearn.utils import resample
from sklearn.preprocessing import MultiLabelBinarizer
from imblearn.over_sampling import SMOTE
import numpy as np
from collections import Counter
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import numpy as np
from collections import Counter
from scipy.stats import zscore
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd


sys.path.append('../')
from utils import baseline_utils


SWATCHES_MAPPING = {
    1: [246, 237, 228],
    2: [243, 231, 219],
    3: [247, 234, 208],
    4: [234, 218, 186],
    5: [215, 189, 150],
    6: [160, 126, 86],
    7: [130, 92, 67],
    8: [96, 65, 52],
    9: [58, 49, 42],
    10: [41, 36, 32]
}

def align_face_ffhq(img, landmarks, output_size, transform_size, enable_padding):
    """ Aligns a face in an image using FFHQ alignment.

    Args:
        img (PIL.Image): The input image.
        landmarks (dlib.full_object_detection): The facial landmarks detected in the image.
        output_size (int): The size of the output image.
        transform_size (int): The size of the transformed image.
        enable_padding (bool): Whether to enable padding.

    Returns:
        PIL.Image: The aligned image.
    """

    lm = np.array([[p.x, p.y] for p in landmarks.parts()])
    
    # Calculate auxiliary vectors.
    eye_left = np.mean(lm[36:42], axis=0)
    eye_right = np.mean(lm[42:48], axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm[48]
    mouth_right = lm[54]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c0 = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c0 - x - y, c0 - x + y, c0 + x + y, c0 + x - y])
    qsize = np.hypot(*x) * 2

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.Resampling.LANCZOS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), 
            int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), 
            min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), 
           int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), 
           max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    # Final Resize
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.Resampling.LANCZOS)

    return img

def process_images(input_dir, output_dir, output_size, shape_predictor_path, transform_size, enable_padding):
    """ Process images in the input directory and save aligned images to the output directory.

    Args:
        input_dir (str): The input directory containing images to process.
        output_dir (str): The output directory to save aligned images.
        output_size (int): The size of the output image.
        shape_predictor_path (str): The path to the shape predictor model.
        transform_size (int): The size of the transformed image.
        enable_padding (bool): Whether to enable padding.    
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Dlib models
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)

    # Iterate over all image files in the input directory
    for img_path in Path(input_dir).rglob("*.*"):  
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".JPG"}:  # Filter image formats
            continue

        try:
            img = PIL.Image.open(img_path).convert("RGB")
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # Detect face
            detections = detector(img_cv, 1)
            if len(detections) == 0:
                print(f"No faces detected in {img_path}")
                continue

            # Use the first detected face for alignment
            landmarks = predictor(img_cv, detections[0])
            aligned_img = align_face_ffhq(img, landmarks, output_size, transform_size, enable_padding)

            relative_path = img_path.relative_to(input_dir)
            output_path = output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save the aligned image
            aligned_img.save(output_path)
            print(f"Saved aligned image to {output_path}")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

def preprocess_data(data, swatches = False):
    """ Preprocesses the data for training a model.

    Args:
        data (dict): The input data dictionary.

    Returns:
        X (np.array): The feature matrix.
        y (np.array): The target matrix
    """

    # Parse features and targets
    X = []
    y = []

    for key, value in data.items():
        mst_value = value["MST_value"]

        for k, features in value.items():
            if k != "MST_value":


                left_cheek_color = features.get("CSEC_cheek_colors", {}).get("left_cheek_color", [0, 0, 0]) or [0, 0, 0]
                right_cheek_color = features.get("CSEC_cheek_colors", {}).get("right_cheek_color", [0, 0, 0]) or [0, 0, 0]
                left_sclera_color = features.get("sclera_RGB", {}).get("left_sclera_color", [0, 0, 0]) or [0, 0, 0]
                right_sclera_color = features.get("sclera_RGB", {}).get("right_sclera_color", [0, 0, 0]) or [0, 0, 0]
                fiqa_exposure = features.get("FIQA_exposure", {}).get("Prob_Class_0", 0.0) or 0.0
                illumination_uniformity = features.get("OFIQ_illumination_uniformity", 0.0) or 0.0
                
                # target = SWATCHES_MAPPING[mst_value]

                if swatches:
                    target = SWATCHES_MAPPING[mst_value]
                else:
                    target = mst_value
                
                feature_vector = (
                    left_cheek_color
                    + right_cheek_color
                    + left_sclera_color
                    + right_sclera_color
                    + [fiqa_exposure, illumination_uniformity]
                )
                X.append(feature_vector)
                y.append(target)
                

    X = np.array(X)
    y = np.array(y)

    return X, y

def save_results(output_file, model_name, train_accuracy, test_accuracy, report, other_metrics=None):
    """ Saves the classification results to a file.

    Args:
        output_file (str): The output file path.
        model_name (str): The name of the model.
        accuracy (float): The accuracy of the model.
        report (str): The classification report.
    """

    # Append results to the output file
    with open(output_file, "a") as f:  
        f.write(f"Model: {model_name}\n")
        f.write(f"Train accuracy: {train_accuracy:.4f}\n")
        f.write(f"Test accuracy: {test_accuracy:.4f}\n\n")
        if other_metrics:
            f.write(other_metrics)
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n" + "="*50 + "\n")  # Separator between models

    print(f"Results for {model_name} saved to {output_file}")


def remove_testing_images_from_training(training_data, sampled_testing_paths_set):
    """ Removes the testing images from the training data.

    Args:
        training_data (dict): The training data dictionary.
        sampled_testing_paths_set (set): The set of sampled testing paths.

    Returns:
        dict: The filtered training data.
    """

    filtered_training_data = {}
    
    for subject, images in training_data.items():
        filtered_training_data[subject] = {}
        for image_name, image_data in images.items():
            if image_name != "MST_value":  
                path = f"{subject}/{image_name}"

                # If the image is not in the sampled testing paths, keep it in the training data
                if path not in sampled_testing_paths_set:
                    filtered_training_data[subject][image_name] = image_data
            else:
                # Keep the MST_value entry
                filtered_training_data[subject]["MST_value"] = images["MST_value"]
    
    return filtered_training_data

def filter_testing_data(testing_data, sampled_testing_paths_set):
    """ Filters the testing data to only include the sampled testing paths.

    Args:
        testing_data (_type_): _description_
        sampled_testing_paths_set (_type_): _description_

    Returns:
        dict: The filtered testing data.
    """

    filtered_testing_data = {}
    
    for subject, images in testing_data.items():
        filtered_testing_data[subject] = {}
        for image_name, image_data in images.items():
            if image_name != "MST_value":  # Skip the MST_value entry
                path = f"{subject}/{image_name}"
                
                # If the image is in the sampled testing paths, keep it
                if path in sampled_testing_paths_set:
                    filtered_testing_data[subject][image_name] = image_data
            else:
                # Keep the MST_value entry
                filtered_testing_data[subject]["MST_value"] = images["MST_value"]
    
    return filtered_testing_data

def oversampling_SMOTE(X, y):
    class_counts = Counter(y)

    # Identify minority classes with very few samples
    minority_classes = [cls for cls, count in class_counts.items() if count < 2]

    # Duplicate samples for minority classes
    for cls in minority_classes:
        mask = y == cls
        X = np.concatenate([X, X[mask]])
        y = np.concatenate([y, y[mask]])
        
    # Apply SMOTE to balance the classes
    smote = SMOTE(k_neighbors=1)
    X, y = smote.fit_resample(X, y)

    return X,y

def oversampling_SMOTE_multilabel(X, y):
    """Oversample the minority classes using SMOTE for multi-label classification.

    Args:
        X (np.array): Feature matrix.
        y (np.array): Multi-label target matrix.

    Raises:
        ValueError: If the sample count mismatch occurs.

    Returns:
        np.array: The resampled feature matrix.
    """

    X_resampled = None
    y_resampled = []

    # Process each label independently
    for i in range(y.shape[1]):
        y_label = y[:, i]  
        class_counts = Counter(y_label)

        # Identify minority classes with very few samples
        minority_classes = [cls for cls, count in class_counts.items() if count < 2]

        # Duplicate samples for minority classes
        X_augmented, y_label_augmented = X, y_label
        for cls in minority_classes:
            mask = (y_label == cls)
            if mask.sum() > 0:  # Ensure the mask is valid
                X_augmented = np.concatenate([X_augmented, X[mask]], axis=0)
                y_label_augmented = np.concatenate([y_label_augmented, y_label[mask]], axis=0)

        # Apply SMOTE to the current label
        smote = SMOTE(k_neighbors=1)
        X_res, y_label_res = smote.fit_resample(X_augmented, y_label_augmented)

        # Handle first iteration
        if X_resampled is None:
            X_resampled = X_res
            y_resampled = y_label_res[:, np.newaxis]
        else:
            if X_res.shape[0] != X_resampled.shape[0]:
                raise ValueError("Mismatch in sample count after resampling!")
            
            y_resampled = np.concatenate([y_resampled, y_label_res[:, np.newaxis]], axis=1)

    return X_resampled, y_resampled



def remove_outliers(X,y):
    """Remove outliers from the feature matrix X and the target matrix y.

    Args:
        X (np,array): The feature matrix.
        y (np.array): The target matrix.

    Returns:
        np.array: The feature matrix without outliers.
        np.array: The target matrix without outliers.
    """

    # Calculate Z-scores for each feature (column)
    z_scores = np.abs(zscore(X))

    threshold = 3
    outliers = np.where(z_scores > threshold)
    outlier_indices = np.unique(outliers[0])  # unique rows with outliers

    # Remove outliers from the feature matrix X and the target matrix y
    X = np.delete(X, outlier_indices, axis=0)
    y = np.delete(y, outlier_indices, axis=0)

    print(f"Outliers detected: {len(outlier_indices)}")
    print(f"Remaining samples: {X.shape[0]}")

    return X, y

def prepare_data_with_split(data_path, swatches = False):
    """ Prepare the data for training and testing by splitting the data into training and testing sets.
    
    Args:
        data_path (str): The path to the data file.
        swatches (bool): Whether to use swatches as the target labels (default

    Returns:
        np.array: The training feature matrix.
        np.array: The testing feature matrix.
        np.array: The training target matrix.
        np.array: The testing target matrix.
    """

    with open(data_path) as f:
        data = json.load(f)

    X_train, y_train = preprocess_data(data, swatches)

    X_train, y_train = remove_outliers(X_train, y_train)

    if swatches:
        X_train, y_train = oversampling_SMOTE_multilabel(X_train, y_train)
    else:
        X_train, y_train = oversampling_SMOTE(X_train, y_train)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    X_train, X_test, y_train, y_test = preprocess_data_with_imputation_and_scaling(X_train, X_test, y_train, y_test)

    print("prepare_data_with_split ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return X_train, X_test, y_train, y_test

def prepare_data_with_original(training_data_path, testing_data_path, nr_of_test_images, swatches = False):
    """ Prepare the data for training and testing by splitting the data into training and testing sets.

    Args:
        training_data_path (str): The path to the training data file.
        testing_data_path (str): The path to the testing data file.
        nr_of_test_images (int): The number of test images to sample.
        swatches (bool): Whether to use swatches as the target labels (default: False).

    Returns:
        np.array: The training feature matrix.
    """

    with open(training_data_path) as f:
        training_data = json.load(f)

    with open(testing_data_path) as f:
        testing_data = json.load(f)

    testing_paths = []
    for subject, images in testing_data.items():
        for image_name, image_data in images.items():
            if image_name != "MST_value": 
                testing_paths.append(f"{subject}/{image_name}")

    random.seed(42)  
    sampled_testing_paths = random.sample(testing_paths, nr_of_test_images)

    sampled_testing_paths_set = set(sampled_testing_paths)

    filtered_training_data = remove_testing_images_from_training(training_data, sampled_testing_paths_set)
    filtered_testing_data = filter_testing_data(testing_data, sampled_testing_paths_set)

    X_train, y_train = preprocess_data(filtered_training_data, swatches)
    X_train, y_train = remove_outliers(X_train, y_train)
    # if swatches:
    #     X_train, y_train = oversampling_SMOTE_multilabel(X_train, y_train)
    # else:
    #     X_train, y_train = oversampling_SMOTE(X_train, y_train)

    X_test, y_test = preprocess_data(filtered_testing_data, swatches)
    X_test, y_test = remove_outliers(X_test, y_test)
    # if swatches:
    #     X_test, y_test = oversampling_SMOTE_multilabel(X_test, y_test)
    # else:
    #     X_test, y_test = oversampling_SMOTE(X_test, y_test)

    X_train, X_test, y_train, y_test = preprocess_data_with_imputation_and_scaling(X_train, X_test, y_train, y_test)

    return X_train, X_test, y_train, y_test

def prepare_data_with_MST(training_data_path, testing_data_path, swatches = False):
    """ Prepare the data for training and testing by splitting the data into training and testing sets.

    Args:
        training_data_path (str): The path to the training data file.
        testing_data_path (str): The path to the testing data file.
        swatches (bool): Whether to use swatches as the target labels (default: False).

    Returns:
        np.array: The training feature matrix.
    """
    with open(training_data_path) as f:
        training_data = json.load(f)

    with open(testing_data_path) as f:
        testing_data = json.load(f)

    X_train, y_train = preprocess_data(training_data, swatches)
    X_train, y_train = remove_outliers(X_train, y_train)
    # if swatches:
    #     X_train, y_train = oversampling_SMOTE_multilabel(X_train, y_train)
    # else:
    #     X_train, y_train = oversampling_SMOTE(X_train, y_train)

    X_test, y_test = preprocess_data(testing_data, swatches)
    X_test, y_test = remove_outliers(X_test, y_test)
    # if swatches:
    #     X_test, y_test = oversampling_SMOTE_multilabel(X_test, y_test)
    # else:
    #     X_test, y_test = oversampling_SMOTE(X_test, y_test)
    

    X_train, X_test, y_train, y_test = preprocess_data_with_imputation_and_scaling(X_train, X_test, y_train, y_test)

    return X_train, X_test, y_train, y_test


def SMOTE_fitting(X, y):
    """ Balance the classes using SMOTE.

    Args:
        X (np.array): Feature matrix.
        y (np.array): Target matrix.

    Returns:
        np.array: The resampled feature matrix.
        np.array: The resampled target matrix.
    """
    
    class_counts = Counter(y)

    # Identify minority classes with very few samples
    minority_classes = [cls for cls, count in class_counts.items() if count < 2]

    # Duplicate samples for minority classes
    for cls in minority_classes:
        mask = y == cls
        X = np.concatenate([X, X[mask]])
        y = np.concatenate([y, y[mask]])
        
    # Apply SMOTE to balance the classes
    smote = SMOTE(k_neighbors=1)
    X, y = smote.fit_resample(X, y)

    return X, y

def generate_synthetic_data(X, y, num_samples=100, noise_level=0.01):
    """Generate synthetic data by adding Gaussian noise to the original data.

    Args:
        X (np.array): Feature matrix.
        y (np.array): Target matrix.
        num_samples (int): Number of synthetic samples to generate.
        noise_level (float): Standard deviation of the Gaussian noise.
    Returns:
        np.array: The synthetic feature matrix.
        np.array: The synthetic target matrix.
    """
    synthetic_X = []
    synthetic_y = []
    for _ in range(num_samples):
        idx = np.random.randint(0, len(X))  
        noise = np.random.normal(0, noise_level, size=X.shape[1])  
        synthetic_X.append(X[idx] + noise)
        synthetic_y.append(y[idx])
    return np.vstack([X, np.array(synthetic_X)]), np.vstack([y, np.array(synthetic_y)])


def generate_cluster_based_samples(X, y, threshold = 40):
    """ Generate synthetic samples by oversampling sparse clusters.
    

    Args:
        X (_type_): Feature matrix.
        y (_type_): Target matrix.
        threshold (_type_): Minimum samples per cluster to avoid oversampling.

    Returns:
        np.array: The oversampled feature matrix.
        np.array: The oversampled target matrix.
    """
    # Apply clustering to identify groups
    kmeans = KMeans(n_clusters=10, random_state=42)
    clusters = kmeans.fit_predict(X)

    # Oversample sparse clusters
    cluster_counts = Counter(clusters)
    oversampled_X = [X]
    oversampled_y = [y]

    for cluster_id, count in cluster_counts.items():
        if count < threshold:
            mask = clusters == cluster_id
            oversampled_X.append(X[mask])
            oversampled_y.append(y[mask])

    # Concatenate all parts to form the final oversampled dataset
    X = np.concatenate(oversampled_X, axis=0)
    y = np.concatenate(oversampled_y, axis=0)

    return X, y



def balance_regression_data(X, y, target_size=None, n_clusters=10, threshold=40):
    """
    Balance regression data using clustering and oversampling.

    Args:
        X (np.array): Feature matrix.
        y (np.array): Continuous target variables (e.g., RGB values).
        target_size (int): Target number of samples per cluster (default: largest cluster size).
        n_clusters (int): Number of clusters to form.
        threshold (int): Minimum samples per cluster to avoid oversampling.

    Returns:
        X_balanced (np.array): Balanced feature matrix.
        y_balanced (np.array): Balanced target variables.
    """
    # Cluster y to identify groups
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(y)

    # Determine target size per cluster
    cluster_counts = Counter(clusters)
    max_cluster_size = max(cluster_counts.values()) if target_size is None else target_size

    X_balanced = []
    y_balanced = []

    for cluster_id, count in cluster_counts.items():
        mask = clusters == cluster_id
        X_cluster = X[mask]
        y_cluster = y[mask]

        if count < threshold:
            # Oversample smaller clusters by duplicating samples
            num_to_add = max_cluster_size - count
            oversampled_indices = np.random.choice(len(X_cluster), num_to_add, replace=True)
            X_balanced.append(np.concatenate([X_cluster, X_cluster[oversampled_indices]]))
            y_balanced.append(np.concatenate([y_cluster, y_cluster[oversampled_indices]]))
        else:
            # Keep the original data for larger clusters
            X_balanced.append(X_cluster)
            y_balanced.append(y_cluster)

    # Combine all clusters into balanced dataset
    X_balanced = np.vstack(X_balanced)
    y_balanced = np.vstack(y_balanced)

    return X_balanced, y_balanced



def preprocess_data_with_imputation_and_scaling(X_train, X_test, y_train, y_test):
    """ Preprocess the data by imputing missing values and scaling the features.

    Args:
        X_train (np.array): The training feature matrix.
        X_test (np.array): The testing feature matrix.
        y_train (np.array): The training target matrix.
        y_test (np.array): The testing target matrix
        
    Returns:
        np.array: The training feature matrix.
        np.array: The testing feature matrix.
        np.array: The training target matrix.
        np.array: The testing target matrix.
    """

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Step 3: Apply PCA for dimensionality reduction
    # pca = PCA(n_components=0.95)  # Retain 95% of the variance
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)  

    # X_train, y_train = balance_regression_data(X_train, y_train, 60)
    # X_test, y_test = balance_regression_data(X_test, y_test, 60)
    # X_train, y_train = generate_synthetic_data(X_train, y_train, num_samples=100)
    # X_test, y_test = generate_synthetic_data(X_test, y_test, num_samples=100)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Oversample minority classes with RandomOverSampler
    # ros = RandomOverSampler(random_state=42)
    # X_train, y_train = ros.fit_resample(X_train, y_train

    return X_train, X_test, y_train, y_test


def train_model_classification(model, experiment, output_file, X_train, X_test, y_train, y_test, swatches = False):
    """ Train a classification model and evaluate its performance.

    Args:
        model (object): The classification model.
        experiment (str): The name of the experiment.
        output_file (str): The output file path.
        X_train (np.array): The training feature matrix.
        X_test (np.array): The testing feature matrix.
        y_train (np.array): The training target matrix.
        y_test (np.array): The testing target matrix.
        swatches (bool): Whether to use swatches as the target labels (default: False).
    """

    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    print(f"Random Forest {experiment}, Test accuracy: {accuracy_score(y_test, y_pred_test)}, Train Accuracy: {accuracy_score(y_train, y_pred_train)}")
    
    if swatches:
        y_train, y_pred_train = RGB_predictions_to_MST(y_train, y_pred_train)
        y_test, y_pred_test = RGB_predictions_to_MST(y_test, y_pred_test)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    report_log_reg = classification_report(y_test, y_pred_test)
    save_results(output_file, type(model).__name__, train_accuracy, test_accuracy, report_log_reg)



    # Plot feature importance

    permutation_importance_path =  os.path.dirname(output_file) + "/permutation_importance"
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

    # Plot the importance scores
    plt.figure(figsize=(10, 6))
    # plt.title(f"Permutation Importance for {type(model).__name__}")
    plt.barh(range(X_train.shape[1]), result.importances_mean, align='center',  color='#52238d', height = 0.6)
    plt.yticks(range(X_train.shape[1]), X_train.columns if hasattr(X_train, 'columns') else np.arange(X_train.shape[1]))
    plt.xlabel('Permutation Importance')
    plt.xlim(-0.5, 0.5)

    plt.savefig(os.path.join(permutation_importance_path, f"{experiment}_{type(model).__name__}_permutation_importance.png"))
    # plt.show()

    # Plot confusion matrix

    confusion_matrix_path =  os.path.dirname(output_file) + "/confusion_matrix"

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_test, labels=range(1, 10), normalize="true")
    conf_matrix = np.round(conf_matrix, 2) 

    display_labels = [str(i) for i in range(1, 10)]  

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=display_labels)
    disp.plot(cmap="Purples")
    # plt.title("Confusion Matrix: Ground Truth vs Predicted MST")
    plt.savefig(os.path.join(confusion_matrix_path, f"{experiment}_{type(model).__name__}_normalize.png"))
    # plt.show()

    conf_matrix = confusion_matrix(y_test, y_pred_test, labels=range(1, 10))
    conf_matrix = np.round(conf_matrix, 2) 

    # Display the confusion matrix
    display_labels = [str(i) for i in range(1, 10)]  

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=display_labels)
    disp.plot(cmap="Purples")
    # plt.title("Confusion Matrix: Ground Truth vs Predicted MST")
    plt.savefig(os.path.join(confusion_matrix_path, f"{experiment}_{type(model).__name__}.png"))
    # plt.show()




def train_model_regression(model, experiment, output_file, X_train, X_test, y_train, y_test, swatches=False):
    """ Train a regression model and evaluate its performance.

    Args:
        model (object): The regression model.
        experiment (str): The name of the experiment.
        output_file (str): The output file path.
        X_train (np.array): The training feature matrix.
        X_test (np.array): The testing feature matrix.
        y_train (np.array): The training target matrix.
        y_test (np.array): The testing target matrix.
        swatches (bool): Whether to use swatches as the target labels (default: False).
    """
    # Fit the model
    model.fit(X_train, y_train)

    # Predict on the test and train set
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    # Calculate Regression Metrics (MSE, MAE, R2)
    mse_test = mean_squared_error(y_test, y_pred_test)
    mse_train = mean_squared_error(y_train, y_pred_train)
    
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    
    r2_test = r2_score(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)

    # Print the regression evaluation metrics
    # print(f"Random Forest {experiment}, Test MSE: {mse_test}, Train MSE: {mse_train}")
    # print(f"Random Forest {experiment}, Test MAE: {mae_test}, Train MAE: {mae_train}")
    # print(f"Random Forest {experiment}, Test R2: {r2_test}, Train R2: {r2_train}")

    # If swatches are enabled, map predictions back to MST values
    if swatches:
        y_train, y_pred_train = RGB_predictions_to_MST(y_train, y_pred_train)
        y_test, y_pred_test = RGB_predictions_to_MST(y_test, y_pred_test)

        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)

        train_accuracy_MST_1 = calculate_accuracy_with_tolerance(y_train, y_pred_train, tolerance=1)
        test_accuracy_MST_1 = calculate_accuracy_with_tolerance(y_test, y_pred_test, tolerance=1)

        train_accuracy_MST_2 = calculate_accuracy_with_tolerance(y_train, y_pred_train, tolerance=2)
        test_accuracy_MST_2 = calculate_accuracy_with_tolerance(y_test, y_pred_test, tolerance=2)

        report_log_reg = classification_report(y_test, y_pred_test)

        # MSE, MAE, R2 string
        metrics_string = f"Test MSE: {mse_test}, Train MSE: {mse_train}\n, Test MAE: {mae_test}, Train MAE: {mae_train}\n, Test R2: {r2_test}, Train R2: {r2_train}\n, Test Accuracy: {test_accuracy}, Train Accuracy: {train_accuracy}\n, Test Accuracy MST ±1: {test_accuracy_MST_1}, Train Accuracy MST ±1: {train_accuracy_MST_1}\n, Test Accuracy MST ±2: {test_accuracy_MST_2}, Train Accuracy MST ±2: {train_accuracy_MST_2}\n"
        save_results(output_file, type(model).__name__, train_accuracy, test_accuracy, report_log_reg, metrics_string)

        # Plot confusion matrix

        confusion_matrix_path =  os.path.dirname(output_file) + "/confusion_matrix_lgb_regressor"

        # Compute the confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred_test, labels=range(1, 10), normalize="true")
        conf_matrix = np.round(conf_matrix, 2) 
        # Display the confusion matrix
        display_labels = [str(i) for i in range(1, 10)]  

        # Display the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=display_labels)
        disp.plot(cmap="Purples")
        # plt.title("Confusion Matrix: Ground Truth vs Predicted MST")
        plt.savefig(os.path.join(confusion_matrix_path, f"{experiment}_{type(model).__name__}_normalize.png"))
        # plt.show()



        # Compute the confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred_test, labels=range(1, 10))
        conf_matrix = np.round(conf_matrix, 2)  
        # Display the confusion matrix
        display_labels = [str(i) for i in range(1, 10)]  

        # Display the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=display_labels)
        disp.plot(cmap="Purples")
        # plt.title("Confusion Matrix: Ground Truth vs Predicted MST")
        plt.savefig(os.path.join(confusion_matrix_path, f"{experiment}_{type(model).__name__}.png"))
        # plt.show()



def RGB_predictions_to_MST(y_true, y_pred):
    """ Convert RGB predictions to MST labels.

    Args:
        y_true (np.array): The true RGB labels.
        y_pred (np.array): The predicted RGB labels
    Returns:
        np.array: The true MST labels.
        np.array: The predicted MST labels
    """


    REVERSE_SWATCHES_MAPPING = {tuple(value): key for key, value in SWATCHES_MAPPING.items()}

    mst_orb_dir = '/home/dasec-notebook/Thesis/Datasets/MST Orbs/orbs'

    mst_orb_salient_colors = {}
    for filename in os.listdir(mst_orb_dir):
        if filename.endswith(".png"): 
            image_path = os.path.join(mst_orb_dir, filename)
            salient_colors = baseline_utils.extract_salient_colors(image_path)
            mst_orb_salient_colors[filename] = salient_colors

    formatted_y_true = []
    formatted_y_pred = []

    for i, y in enumerate(y_pred):
        true_MST = REVERSE_SWATCHES_MAPPING.get(tuple(y_true[i]), None)
        if true_MST is None:
            true_MST, _ = baseline_utils.calculate_best_mst_orb(y_true[i], mst_orb_salient_colors)
            true_MST = int(true_MST)
            # print(f"True MST not found for RGB: {y_true[i]}, mapped to {true_MST}")
        formatted_y_true.append(true_MST)
        
        # Map predicted RGB to MST label
        predicted_MST, _ = baseline_utils.calculate_best_mst_orb(y_pred[i], mst_orb_salient_colors)
        formatted_y_pred.append(int(predicted_MST))

    return formatted_y_true, formatted_y_pred

def RGB_predictions_euclidian_to_MST(y_true, y_pred):
    """ Convert RGB predictions to MST labels using Euclidean distance.

    Args:
        y_true (_type_): The true RGB labels.
        y_pred (_type_): The predicted

    Returns:
        _type_: The true MST labels.
        _type_: The predicted MST labels
    """
    # Create reverse mapping from RGB to MST index for easy lookup
    REVERSE_SWATCHES_MAPPING = {tuple(value): key for key, value in SWATCHES_MAPPING.items()}

    # Calculate Euclidean distance between RGB values
    def euclidean_distance(rgb1, rgb2):
        return np.sqrt(np.sum((np.array(rgb1) - np.array(rgb2)) ** 2))

    formatted_y_true = []
    formatted_y_pred = []

    for i, y in enumerate(y_pred):
        true_MST = REVERSE_SWATCHES_MAPPING.get(tuple(y_true[i]), None)
        formatted_y_true.append(true_MST)
        
        # Find the closest MST swatch for the predicted RGB value
        closest_MST = None
        min_distance = float('inf')

        for mst_index, swatch_rgb in SWATCHES_MAPPING.items():
            distance = euclidean_distance(y_pred[i], swatch_rgb)
            if distance < min_distance:
                min_distance = distance
                closest_MST = mst_index
        
        formatted_y_pred.append(closest_MST)

    return formatted_y_true, formatted_y_pred


def calculate_accuracy_with_tolerance(y_true, y_pred, tolerance=2):
    """
    Calculates the accuracy based on whether the predicted values are within a tolerance of the true values.

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted values.
        tolerance (int): The tolerance range (default is ±2).

    Returns:
        float: The percentage of correct predictions within the tolerance.
    """
    correct_predictions = 0
    total_predictions = len(y_true)

    for true, pred in zip(y_true, y_pred):
        if abs(true - pred) <= tolerance:
            correct_predictions += 1

    accuracy = (correct_predictions / total_predictions)

    return accuracy

from sklearn.preprocessing import label_binarize


def plot_multiclass_roc_curve(model, X_train, X_test, y_train, y_test):
    """ Plot the ROC curve for a multiclass classification model.

    Args:
        model: The trained classification model.
        X_train: The training feature matrix.
        X_test: The testing feature matrix.
        y_train: The training target matrix.
        y_test: The testing target matrix.

    Raises:
        ValueError: If the model does not support predict_proba or decision_function.
    """

    n_classes = len(np.unique(y_train))

    # Binarize the output labels
    y_train_bin = label_binarize(y_train, classes=range(1, n_classes + 1))
    y_test_bin = label_binarize(y_test, classes=range(1, n_classes + 1))

    model.fit(X_train, y_train)
    
    # Check if model supports probabilities (predict_proba), else use decision_function
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)  
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)  
    else:
        raise ValueError("Model must support predict_proba or decision_function for ROC curve.")

    # Plot ROC curve for each class
    plt.figure(figsize=(10, 8))

    # Colors for each class
    colors = plt.cm.get_cmap("tab10", n_classes)

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors(i), lw=2, label=f'Class {i+1} (AUC = {roc_auc:.2f})')

    # Plot diagonal line (random guess)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)

    print(type(model).__name__,)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # plt.title("Multiclass ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(model, X_train, y_train, X_test, y_test, classes=None, normalize=False):
    """  Plots the confusion matrix for a classification model.

    Args:
        model: The trained classification model.
        X_train: The training feature matrix.
        y_train: The training target matrix.
        X_test: The testing feature matrix.
        y_test: The testing target matrix.
        classes: The class labels.
        normalize: Whether to normalize the confusion matrix (default: False).
    """

    model.fit(X_train, y_train)

    classes = np.unique(y_train)

    y_pred = model.predict(X_test)
    
    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Normalize the confusion matrix if specified
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create a heatmap of the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues", 
                xticklabels=classes, yticklabels=classes, cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

from matplotlib.patches import Ellipse

def plot_confidence_ellipse(ax, X, color, label=None):
    """ Plots a confidence ellipse for the given data points

    Args:
        ax: The axis to plot the ellipse on
        X: The data points
        color: The color of the ellipse
        label: The label for the ellipse
    """
    if X.shape[0] < 2:  # Can't compute covariance with fewer than 2 points
        return
    cov = np.cov(X, rowvar=False) 
    mean = np.mean(X, axis=0)  
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * np.sqrt(eigenvalues[:2]) 
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, color=color, alpha=0.3, label=label)
    ax.add_patch(ellipse)