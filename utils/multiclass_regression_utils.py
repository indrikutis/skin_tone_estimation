import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import random
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from collections import Counter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scipy.stats import zscore
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from sklearn.metrics import log_loss


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


def RGB_predictions_to_MST(y_true, y_pred):
    """ Maps RGB predictions to MST labels.

    Args:
        y_true (list): List of true RGB values.
        y_pred (list): List of predicted RGB values
        
    Returns:
        formatted_y_true (list): List of true MST labels.
        formatted_y_pred (list): List of predicted MST labels.
    """

    REVERSE_SWATCHES_MAPPING = {tuple(value): key for key, value in SWATCHES_MAPPING.items()}

    mst_orb_dir = '/home/dasec-notebook/Thesis/Datasets/MST Orbs/orbs'

    mst_orb_salient_colors = {}
    for filename in os.listdir(mst_orb_dir):
        if filename.endswith(".png"):  # Assumes MST orbs are PNG images
            image_path = os.path.join(mst_orb_dir, filename)
            salient_colors = baseline_utils.extract_salient_colors(image_path)
            mst_orb_salient_colors[filename] = salient_colors


    # Process y_true and y_pred
    formatted_y_true = []
    formatted_y_pred = []

    for i, y in enumerate(y_pred):
        # Map true RGB to MST label
        true_MST = REVERSE_SWATCHES_MAPPING.get(tuple(y_true[i]), None)
        formatted_y_true.append(true_MST)
        
        # Map predicted RGB to MST label
        predicted_MST, _ = baseline_utils.calculate_best_mst_orb(y_pred[i], mst_orb_salient_colors)
        formatted_y_pred.append(int(predicted_MST))

    return formatted_y_true, formatted_y_pred

def accuracy_score_MST(y_true, y_pred):
    """ Calculate the accuracy of the model based on the MST labels.

    Args:
        y_true (list): List of true MST labels.
        y_pred (list): List of predicted MST labels
    """

    # Ensure both lists have the same length
    assert len(y_true) == len(y_pred), "Lists must have the same length"

    # Initialize counters
    total = len(y_true)
    exact_matches = 0
    plus_minus_1_matches = 0
    plus_minus_2_matches = 0

    # Calculate accuracies
    for true, predicted in zip(y_true, y_pred):
        if true == predicted:
            exact_matches += 1
        if abs(true - predicted) <= 1:
            plus_minus_1_matches += 1
        if abs(true - predicted) <= 2:
            plus_minus_2_matches += 1

    # Compute accuracy percentages
    exact_accuracy = exact_matches / total * 100
    plus_minus_1_accuracy = plus_minus_1_matches / total * 100
    plus_minus_2_accuracy = plus_minus_2_matches / total * 100

    # Print results
    print(f"Exact Match Accuracy: {exact_accuracy:.2f}%")
    print(f"±1 Match Accuracy: {plus_minus_1_accuracy:.2f}%")
    print(f"±2 Match Accuracy: {plus_minus_2_accuracy:.2f}%")


def preprocess_data(data):
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

        # print(key)
        for k, features in value.items():
            if k != "MST_value":
                # print(k, v)

                # Extract the nested dictionary (e.g., "DSC_0703.JPG")
                # nested_key = [k for k in value.keys() if k != "MST_value"][0]  # Find the nested dictionary key
                # features = value[nested_key]  # Access the nested dictionary
                # print(features)

                left_cheek_color = features.get("CSEC_cheek_colors", {}).get("left_cheek_color", [0, 0, 0]) or [0, 0, 0]
                right_cheek_color = features.get("CSEC_cheek_colors", {}).get("right_cheek_color", [0, 0, 0]) or [0, 0, 0]
                left_sclera_color = features.get("sclera_RGB", {}).get("left_sclera_color", [0, 0, 0]) or [0, 0, 0]
                right_sclera_color = features.get("sclera_RGB", {}).get("right_sclera_color", [0, 0, 0]) or [0, 0, 0]
                fiqa_exposure = features.get("FIQA_exposure", {}).get("Prob_Class_0", 0.0) or 0.0
                illumination_uniformity = features.get("OFIQ_illumination_uniformity", 0.0) or 0.0
                
                # target = SWATCHES_MAPPING[mst_value]
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

    X, y = remove_outliers_z_score(X, y)

    return X, y

def prepare_data_with_split(data, test_size=0.2):
    """ Prepares the data for training a model.

    Args:
        data (str): The path to the input data file.
        test_size (float): The size of the test set.

    Returns:
        X_train (np.array): The training feature matrix.
        X_test (np.array): The testing feature matrix.
        y_train (np.array): The training target matrix.
        y_test (np.array): The testing target matrix.
        num_classes (int): The number of classes in the dataset
    """

    with open(data, 'r') as f:
        data = json.load(f)

    X, y = preprocess_data(data)
    num_classes = len(np.unique(y))

    X, y = oversampling_normalization(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test, num_classes

def prepare_data_two_sources(train_data, test_data):
    """ Prepares the data for training a model using two separate sources.

    Args:
        train_data (str): The path to the training data file.
        test_data (str): The path to the testing data file.

    Returns:
        X_train (np.array): The training feature matrix.
        X_test (np.array): The testing feature matrix.
        y_train (np.array): The training target matrix.
        y_test (np.array): The testing target matrix.
        num_classes (int): The number of classes in the dataset
    """

    with open(train_data, 'r') as f:
        train_data = json.load(f)

    with open(test_data, 'r') as f:
        test_data = json.load(f)

    X_train, y_train = preprocess_data(train_data)
    X_test, y_test = preprocess_data(test_data)

    # Find common classes
    train_classes = set(np.unique(y_train))
    test_classes = set(np.unique(y_test))
    common_classes = train_classes.intersection(test_classes)

    # Filter training data
    train_mask = np.isin(y_train, list(common_classes))
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]

    # Filter testing data
    test_mask = np.isin(y_test, list(common_classes))
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    num_classes = len(common_classes)

    # Oversample and normalize the data
    X_train, y_train = oversampling_normalization(X_train, y_train)
    X_test, y_test = oversampling_normalization(X_test, y_test)

    return X_train, X_test, y_train, y_test, num_classes

def remove_outliers_z_score(X,y):
    """ Removes outliers from the feature matrix X and the target matrix y using Z-scores.

    Args:
        X (np.array): The feature matrix.
        y (np.array): The target matrix.

    Returns:
        X (np.array): The feature matrix without outliers.
        y (np.array): The target matrix without outliers.
    """

    z_scores = np.abs(zscore(X))

    # Define a threshold for detecting outliers
    threshold = 3

    # Find samples where any feature has a Z-score greater than the threshold
    outliers = np.where(z_scores > threshold)
    outlier_indices = np.unique(outliers[0])  # unique rows with outliers

    # Remove outliers from the feature matrix X and the target matrix y
    X = np.delete(X, outlier_indices, axis=0)
    y = np.delete(y, outlier_indices, axis=0)

    print(f"Outliers detected: {len(outlier_indices)}")
    print(f"Remaining samples: {X.shape[0]}")

    return X, y

def oversampling_normalization(X, y):
    """ Oversamples the minority classes and normalizes the feature matrix.

    Args:
        X (np.array): The feature matrix.
        y (np.array): The target matrix.

    Returns:
        X (np.array): The normalized feature matrix.
        y (np.array): The target matrix with one-hot encoding.
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

    # Normalize features (RGB values are in 0-255, normalize to 0-1)
    X = np.array(X) / 255.0
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert targets to one-hot encoding
    y = np.array(y).reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False) 
    y = encoder.fit_transform(y)

    return X, y

def model_training(X_train, X_test, y_train, y_test, num_classes, experiment, output_file):
    """ Trains a neural network model for multiclass regression.

    Args:
        X_train (np.array): The training feature matrix.
        X_test (np.array): The testing feature matrix.
        y_train (np.array): The training target matrix.
        y_test (np.array): The testing target matrix.
        num_classes (int): The number of classes in the dataset.
        experiment (str): The name of the experiment.
        output_file (str): The path to the output file.

    Returns:
        model (Sequential): The trained neural network model.
        history (History): The training history of the model.
    """

    # Define the neural network model
    model = Sequential([
        Dense(128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(momentum=0.99),
        Dropout(0.2),
        Dense(64, activation='relu'),
        BatchNormalization(momentum=0.99),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(momentum=0.99),
        Dense(num_classes, activation='softmax')
    ])


    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)


    # Train the model
    history = model.fit(X_train, y_train, 
                        validation_data=(X_test, y_test),
                        # class_weight=class_weights,
                        epochs=80, 
                        batch_size=32,
                        callbacks=[early_stopping, lr_scheduler])

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)  # Evaluate on training data

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predictions to class labels

    log_loss_value = log_loss(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # Get classification report
    report = classification_report(np.argmax(y_test, axis=1), y_pred_classes)
    # print(classification_report(np.argmax(y_test, axis=1), y_pred_classes))

    save_results(output_file, experiment, train_loss, train_accuracy, test_loss, test_accuracy, log_loss_value, mae, mse, report)

    return model, history

def save_results(output_file, model_name, train_loss, train_accuracy, test_loss, test_accuracy, log_loss_value, mae, mse, report, other_metrics=None):
    """ Saves the classification results to a file.

    Args:
        output_file (str): The output file path.
        model_name (str): The name of the model.
        accuracy (float): The accuracy of the model.
        report (str): The classification report.
    """

    # Apped results to the output file
    with open(output_file, "a") as f:  
        f.write(f"Model: {model_name}\n")
        f.write(f"Train loss: {train_loss:.4f}\n")
        f.write(f"Train accuracy: {train_accuracy:.4f}\n")
        f.write(f"Test loss: {test_loss:.4f}\n")
        f.write(f"Test accuracy: {test_accuracy:.4f}\n\n")
        f.write(f"Log Loss: {log_loss_value:.4f}\n")
        f.write(f"Mean Absolute Error: {mae:.4f}\n")
        f.write(f"Mean Squared Error: {mse:.4f}\n")
        if other_metrics:
            f.write(other_metrics)
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n" + "="*50 + "\n")  # Separator between models

    print(f"Results for {model_name} saved to {output_file}")

def plot_training_history(experiment, history, save_resuts_folder):
    """ Plots the training and validation accuracy and loss.

    Args:
        experiment (str): The name of the experiment.
        history (History): The training history of the model.
        save_resuts_folder (str): The folder to save the
    """

    save_path = f"{save_resuts_folder}/plots/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Get the training and validation accuracy and loss from the history object
    history_accuracy = history.history['accuracy']
    history_val_accuracy = history.history['val_accuracy']
    history_loss = history.history['loss']
    history_val_loss = history.history['val_loss']

    # # Plot training & validation accuracy values
    # plt.figure(figsize=(12, 6))

    # plt.subplot(1, 2, 1)
    # plt.plot(history_accuracy, label='Training Accuracy')
    # plt.plot(history_val_accuracy, label='Validation Accuracy')
    # plt.title('Training and Validation Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()

    # # Plot training & validation loss values
    # plt.subplot(1, 2, 2)
    # plt.plot(history_loss, label='Training Loss')
    # plt.plot(history_val_loss, label='Validation Loss')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()

    # # Show the plots
    # plt.tight_layout()
    # plt.show()

    print("Training and Validation Accuracy")
    # Plot training & validation accuracy values in a separate figure
    plt.figure(figsize=(8, 6))  # Create the first figure
    plt.plot(history_accuracy, label='Training Accuracy')
    plt.plot(history_val_accuracy, label='Validation Accuracy')
    # plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.savefig(save_path + experiment + '_training_validation_accuracy.png')

    plt.show()


    print("Training and Validation Loss")
    # Plot training & validation loss values in another separate figure
    plt.figure(figsize=(8, 6))  # Create the second figure
    plt.plot(history_loss, label='Training Loss')
    plt.plot(history_val_loss, label='Validation Loss')
    # plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.savefig(save_path + experiment + '_training_validation_loss.png')

    plt.show()

