import cv2
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from PIL import Image
import math
from collections import defaultdict
from sklearn.cluster import KMeans
import json

import sys
sys.path.append('/home/dasec-notebook/Thesis/OFIQ-Project/python')

from ofiq_zmq import OfiqZmq


left_cheek_indices = [0, 1, 2, 3, 4, 5, 6]
right_cheek_indices = [26, 27, 28, 29, 30, 31, 32]
left_nose_indices = [55, 56]
right_nose_indices = [58, 59]
left_mouth_indices = [76, 77, 87]
right_mouth_indices = [81, 82, 83]
lower_nose_indices = [56, 57, 58]
left_eye_indices = [60, 61, 62, 63, 64, 65, 66, 67, 96]
right_eye_indices = [68, 69, 70, 71, 72, 73, 74, 75, 97]
middle_nose_indices = [52, 53]


def image_crop(img, left_crop, right_crop, top_crop, bottom_crop, save_path=None):
    """Crop an image using the specified dimensions and save it to a file.

    Args:
        img (ndarray): The image array.
        left_crop (int): The number of pixels to crop from the left.
        right_crop (int): The number of pixels to crop from the right.
        top_crop (int): The number of pixels to crop from the top.
        bottom_crop (int): The number of pixels to crop from the bottom.
        save_path (str): The path to save the cropped image (default is None).

    Returns:
        ndarray: The cropped image array.
    """

    # Step 1: Initial crop
    # left_crop = 2600  # 2100
    # right_crop = width - 2600 # 2400
    # top_crop = 900 # 700
    # bottom_crop = height - 1200 # 850

    # Ensure cropping dimensions are valid
    if left_crop >= right_crop or top_crop >= bottom_crop:
        print("Error: Cropping dimensions are larger than the image size.")
        return

    if save_path is not None:
        cv2.imwrite(save_path, img[top_crop: bottom_crop, left_crop: right_crop])
        print(f"Image saved to {save_path}")


    return img[top_crop: bottom_crop, left_crop: right_crop]


def is_indoor(image_path):
    """Check if an image is taken indoors based on color segmentation.

    Args:
        image_path (str): The path to the image.

    Returns:
        bool: True if the image is taken indoors, False otherwise.
    """
    image = cv2.imread(image_path)

    # Convert to HSV color space for better color segmentation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for grey (indoor) and green/blue (outdoor)
    grey_lower = np.array([0, 0, 70])
    grey_upper = np.array([180, 30, 220])

    # Green/Blue (outdoor) ranges in HSV
    green_lower = np.array([35, 50, 50])
    green_upper = np.array([85, 255, 255])
    blue_lower = np.array([90, 50, 50])
    blue_upper = np.array([130, 255, 255])

    grey_mask = cv2.inRange(hsv_image, grey_lower, grey_upper)
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
    blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)

    # Calculate the proportion of each mask in the image
    grey_ratio = np.sum(grey_mask) / (100 * 100)  
    green_ratio = np.sum(green_mask) / (100 * 100)
    blue_ratio = np.sum(blue_mask) / (100 * 100)

    # Classify as indoor if grey area is very high and green/blue are minimal
    if grey_ratio > 0.7 and green_ratio < 0.1 and blue_ratio < 0.1:
        return True  # Indoor
    else:
        return False  # Outdoor
    

def extract_cheek_patches(landmarks):
    """Extract the coordinates for the left and right cheek patches.

    Args:
        landmarks (ndarray): The facial landmarks array.

    Returns:
        tuple: The coordinates for the left and right cheek patches.
    """

    # Extract coordinates
    left_cheek_points = landmarks[left_cheek_indices]
    right_cheek_points = landmarks[right_cheek_indices]
    left_nose_points = landmarks[left_nose_indices]
    right_nose_points = landmarks[right_nose_indices]
    left_mouth_points = landmarks[left_mouth_indices]
    right_mouth_points = landmarks[right_mouth_indices]
    lower_nose_points = landmarks[lower_nose_indices]
    left_eye_points = landmarks[left_eye_indices]
    right_eye_points = landmarks[right_eye_indices]
    middle_nose_points = landmarks[middle_nose_indices]
 

    # Get bounds for the left cheek patch
    left_cheek_x_min = (np.max(left_cheek_points[:, 0]) + np.min(left_eye_points[:, 0])) / 2
    left_cheek_x_max = np.min(np.concatenate((left_mouth_points[:, 0], left_nose_points[:, 0])))
    left_cheek_y_min = np.min(lower_nose_points[:, 1])
    left_cheek_y_max = np.max(np.concatenate((left_eye_points[:, 1], middle_nose_points[:, 1])))

    # print(left_cheek_x_min, left_cheek_x_max, left_cheek_y_min, left_cheek_y_max)
    
    # Get bounds for the right cheek patch
    
    # Right Cheek Coordinates
    right_cheek_x_max = (np.min(right_cheek_points[:, 0]) + np.max(right_eye_points[:, 0])) / 2
    right_cheek_x_min = np.max(np.concatenate((right_mouth_points[:, 0], right_nose_points[:, 0])))
    right_cheek_y_min = np.min(lower_nose_points[:, 1])
    right_cheek_y_max = np.max(np.concatenate((right_eye_points[:, 1], middle_nose_points[:, 1])))

    left_cheek_coords = (left_cheek_x_min, left_cheek_x_max, left_cheek_y_min, left_cheek_y_max)
    right_cheek_coords = (right_cheek_x_min, right_cheek_x_max, right_cheek_y_min, right_cheek_y_max)

    return left_cheek_coords, right_cheek_coords

def draw_rectangles_on_image(image, left_cheek_coords, right_cheek_coords):
    """Draw rectangles on an image using the specified coordinates.

    Args:
        image (ndarray): The image array.
        left_cheek_coords (tuple): The coordinates for the left cheek patch.
        right_cheek_coords (tuple): The coordinates for the right cheek patch.
    Returns:
        ndarray: The image with rectangles drawn.
    """

    left_cheek_coords = [int(coord) for coord in left_cheek_coords]
    right_cheek_coords = [int(coord) for coord in right_cheek_coords]

    if len(left_cheek_coords) == 4:
        left_cheek_x_min, left_cheek_x_max, left_cheek_y_min, left_cheek_y_max = left_cheek_coords
        
        # Draw the left cheek rectangle
        cv2.rectangle(image, 
                      (left_cheek_x_min, left_cheek_y_min), 
                      (left_cheek_x_max, left_cheek_y_max), 
                      (255, 0, 0), 2) # Blue rectangle for left cheek

    if len(right_cheek_coords) == 4:
        right_cheek_x_min, right_cheek_x_max, right_cheek_y_min, right_cheek_y_max = right_cheek_coords
        
        # Draw the right cheek rectangle
        cv2.rectangle(image, 
                      (right_cheek_x_min, right_cheek_y_min), 
                      (right_cheek_x_max, right_cheek_y_max), 
                      (0, 0, 255), 2)  # Red rectangle for right cheek

    # Return the image with rectangles
    return image


def image_with_rect(img, landmarks):
    """Draw landmarks and rectangles on an image.

    Args:
        img (ndarray): The image array.
        landmarks (ndarray): The facial landmarks array.

    Returns:
        ndarray: The image with landmarks and rectangles drawn.
    """

    # Draw landmarks on the image
    for (x, y) in landmarks: 
        cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)  # Green circles for landmarks

    left_cheek_coords, right_cheek_coords = extract_cheek_patches(landmarks)

    draw_rectangles_on_image(img, left_cheek_coords, right_cheek_coords)
    
    return img 


def get_random_image_paths(base_dir, num_samples=15):
    """Get a list of random image paths from the specified directory.

    Args:
        base_dir (str): The base directory containing the images.
        num_samples (int): The number of random samples to select (default is 15).
    Returns:
        list: List of random image paths.
    """
    image_paths = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                image_paths.append(os.path.join(root, file))
    
    num_samples = min(num_samples, len(image_paths))
    
    # Randomly select the specified number of image paths
    random_image_paths = random.sample(image_paths, num_samples)
    
    return random_image_paths

def display_images(file_path):
    """Display images from a file containing image paths.

    Args:
        file_path (str): The path to the file containing image paths.
    """
    
    with open(file_path, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]


    num_images = len(image_paths)
    cols = math.ceil(math.sqrt(num_images)) 
    rows = math.ceil(num_images / cols)      

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()  

    for i, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path)        
            axes[i].imshow(img)              
            axes[i].axis('off')               
            axes[i].set_title(img_path.split('/')[-1], fontsize=5) 

        except Exception as e:
            print(f"Could not load {img_path}: {e}")
            
            axes[i].axis('off')              

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def calculate_patch_avg_color(img, mask, coords, window_size=10, stride=5):
    """
    Calculate the average RGB color using a sliding window over the specified coordinates,
    and plot each window along with its RGB value.

    Args:
        img (ndarray): The image array.
        coords (tuple): The coordinates of the patch (x_min, x_max, y_min, y_max).
        window_size (int): The size of the sliding window (default is 10).
        stride (int): The stride of the sliding window (default is 5).

    Returns:
        list: List of mean colors for each valid window.
    """
    x_min, x_max, y_max, y_min = (int(coord) for coord in coords)  
    patch_colors = []

    img = np.array(img)
    # Create a copy of the image to draw rectangles
    # img_with_windows = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB).copy()

    # Slide the window over the specified coordinates
    for y in range(y_min, y_max - window_size + 1, stride):
        for x in range(x_min, x_max - window_size + 1, stride):
            
            # Extract the window
            window = img[y:y + window_size, x:x + window_size]
            mask_window = mask[y:y + window_size, x:x + window_size]

            mask_window_squeezed = mask_window.squeeze()  # Shape becomes (10, 10)

            # Select only valid (unmasked) pixels
            valid_pixels = window[mask_window_squeezed == 1]
            if valid_pixels.size > 0:
                mean_color = np.mean(valid_pixels.reshape(-1, 3), axis=0).astype(int)
                patch_colors.append(mean_color)
            
                # Visualize a window with or without mask
                # if np.all(mask_window_squeezed == 1):
                #     cv2.rectangle(img_with_windows, (x, y), (x + window_size, y + window_size), (0, 255, 0), 1)
                # else:
                #     cv2.rectangle(img_with_windows, (x, y), (x + window_size, y + window_size), (255, 0, 0), 1)


    # Calculate the mean color
    if patch_colors:
        avg_color = np.mean(patch_colors, axis=0).astype(int)
    else:
        avg_color = None


    # Display the accumulated patches
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

    # Display the image with all windows highlighted
    # plt.figure(figsize=(4, 4))
    # plt.imshow(img_with_windows)
    # plt.title("Image with Highlighted Windows")
    # plt.axis('off')
    # plt.show()


    return avg_color


def calculate_mean_cheek_color(img, mask, landmarks):
    """
    Calculate the mean RGB color of the left and right cheek patches of an image.
    
    Args:
        img_path (str): The path to the image.
        model_p (str): The path to the landmarks model.

    Returns:
        dict: Dictionary containing the mean color for the left cheek, right cheek, 
              and the combined average color.
    """


    # Extract cheek coordinates
    left_cheek_coords, right_cheek_coords = extract_cheek_patches(landmarks)

    # Calculate average colors for left and right cheeks
    left_cheek_color = calculate_patch_avg_color(img, mask, left_cheek_coords)
    right_cheek_color = calculate_patch_avg_color(img, mask, right_cheek_coords)

    # Calculate overall mean color if both cheeks have valid colors
    if left_cheek_color is not None and right_cheek_color is not None:
        avg_cheek_color = np.mean([left_cheek_color, right_cheek_color], axis=0).astype(int)
    else:
        avg_cheek_color = None

    return {
        "left_cheek_color": left_cheek_color.tolist() if left_cheek_color is not None else None,
        "right_cheek_color": right_cheek_color.tolist() if right_cheek_color is not None else None,
        "avg_cheek_color": avg_cheek_color.tolist() if avg_cheek_color is not None else None
    }



def extract_skin_patch_RGBs(image_paths, output_json_file):

    ofiq_zmq = OfiqZmq('/home/dasec-notebook/Thesis/OFIQ-Project')

    all_patients_data = {}

    for img_path in image_paths:

        # Get the patient ID from the image path
        patient_id = img_path.split('/')[-2] 
        image_filename = os.path.basename(img_path)

        result = ofiq_zmq.process_image(img_path)
        img = result['aligned_face'][0]

        landmarks = np.array([[point.x, point.y] for point in result['aligned_face_landmarks'][1]])
        mask = result['face_occlusion_segmentation_image'][0]

        # Calculate mean cheek colors
        cheek_colors = calculate_mean_cheek_color(img, mask, landmarks)

        if patient_id not in all_patients_data:
            all_patients_data[patient_id] = {}

        # Store cheek color data under the image filename
        all_patients_data[patient_id][image_filename] = cheek_colors

        print(f"Processed image: {img_path}")

    # Save all patients' data to a single JSON file
    with open(output_json_file, 'w') as json_file:
        json.dump(all_patients_data, json_file, indent=4)

    print(f"All data saved to {output_json_file}")


def plot_images_by_mst_category(MST_SUBJECT_MAPPING, base_folder):
    """
    Plots images grouped by MST categories from 1 to 10.

    Args:
        base_folder (str): Path to the base folder containing subject folders.
    """

    mst_images = defaultdict(list)

    # Group images by MST category
    for subject_folder in os.listdir(base_folder):
        if subject_folder in MST_SUBJECT_MAPPING:
            mst_category = MST_SUBJECT_MAPPING[subject_folder]
            subject_path = os.path.join(base_folder, subject_folder)

            for img_file in os.listdir(subject_path):
                img_path = os.path.join(subject_path, img_file)
                if img_file.endswith(('jpg', 'jpeg', 'png')): 
                    mst_images[mst_category].append(img_path)

    # Display images for each MST category
    for mst, images in mst_images.items():
        plt.figure(figsize=(15, 10))
        plt.suptitle(f'MST Category: {mst}', fontsize=16)

        num_images = len(images)
        cols = 12  
        rows = (num_images // cols) + (num_images % cols > 0)  

        # Plot each image in a subplot
        for i, img_path in enumerate(images):
            img = plt.imread(img_path)
            ax = plt.subplot(rows, cols, i + 1)
            ax.imshow(img)
            ax.axis('off') 

            file_name = os.path.basename(img_path) 
            ax.set_title(file_name, fontsize=8, pad=5)  

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        plt.show()



def linear_rgb(rgb):
    """ Convert sRGB to linear RGB values.
    Args:
        rgb (tuple): The sRGB values to convert.

    Returns:
        ndarray: The linear RGB values.
    """
    
    rgb = np.array(rgb) / 255.0
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(linear_rgb_values):
    """ Convert linear RGB values to sRGB.
    Args:
        linear_rgb_values (ndarray): The linear RGB values to convert.

    Returns:
        ndarray: The sRGB values.
    """

    # Scale to [0, 1] range for linear RGB values
    linear_rgb = np.clip(linear_rgb_values / 255.0, 0, 1)
    
    # Apply the sRGB transformation
    srgb = np.where(
        linear_rgb <= 0.0031308,
        linear_rgb * 12.92,
        1.055 * np.power(linear_rgb, 1 / 2.4) - 0.055
    )
    
    # Scale back to [0, 255] and return as integer values
    return np.round(srgb * 255).astype(int)

def extract_salient_colors(image_path, num_colors=3):
    """ Extract the salient colors from an image using K-means clustering.

    Args:
        image_path (str): The path to the image.
        num_colors (int): The number of salient colors to extract (default is 3).

    Returns:
        ndarray: The salient colors extracted from the image
    """

    image = Image.open(image_path).convert("RGBA")
    image_data = np.array(image)
    
    rgb_pixels = image_data[:, :, :3]  
    alpha_channel = image_data[:, :, 3]  
    
    # Mask out transparent pixels (where alpha is 0)
    non_transparent_pixels = rgb_pixels[alpha_channel > 0]  
    
    # Apply K-means clustering to non-transparent pixels
    kmeans = KMeans(n_clusters=num_colors, random_state=0)
    kmeans.fit(non_transparent_pixels)
    
    # Cluster centers represent the salient colors
    salient_colors = kmeans.cluster_centers_.astype(int)  

    return salient_colors

def compute_rmse(measurement, cluster_centers):
    """ Compute the Root Mean Squared Error (RMSE) between a measurement and cluster centers.

    Args:
        measurement (ndarray): The measurement to compare.
        cluster_centers (ndarray): The cluster centers to compare against.

    Returns:
        tuple: The lowest RMSE and the index of the best match.
    """

    errors = np.sqrt(np.mean((cluster_centers - measurement) ** 2, axis=1))
    lowest_rmse = np.min(errors)
    best_match_index = np.argmin(errors)

    return lowest_rmse, best_match_index  


def calculate_best_mst_orb(rgb, mst_orb_salient_colors): 
    """ Calculate the best MST orb for a given RGB color based on the salient colors.

    Args:
        rgb (tuple): The RGB color to match.
        mst_orb_salient_colors (dict): The salient colors for each MST orb.

    Returns:
        tuple: The best MST orb and the lowest RMSE value.    
    """      

    linear_combined_rgb = linear_rgb(rgb)

    best_mst_orb = None
    lowest_rmse = float('inf')

    # Compare against each MST orb's salient colors
    for mst_orb_name, salient_colors in mst_orb_salient_colors.items():
        linear_salient_colors = [linear_rgb(color) for color in salient_colors]

        rmse, _ = compute_rmse(linear_combined_rgb, linear_salient_colors)

        # Check if this orb has the lowest RMSE so far
        if rmse < lowest_rmse:
            lowest_rmse = rmse
            best_mst_orb = mst_orb_name.split('.')[0].split('_')[1] 

    return best_mst_orb, lowest_rmse
