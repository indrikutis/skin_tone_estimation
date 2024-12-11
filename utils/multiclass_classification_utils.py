import os
import numpy as np
import dlib
import PIL.Image
import cv2
import scipy.ndimage
from pathlib import Path
import numpy as np


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

    # Ensure output directory exists
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

            # Recreate the folder structure in the output directory
            relative_path = img_path.relative_to(input_dir)
            output_path = output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save the aligned image
            aligned_img.save(output_path)
            print(f"Saved aligned image to {output_path}")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")


def preprocess_data(data):
    """ Preprocesses the data for multiclass classification.

    Args:
        data (dict): The data dictionary containing the image data.

    Returns:
        np.ndarray: The features array.
    """

    features = []
    targets = []
    
    for subject_id, subject_data in data.items():
        mst_value = subject_data["MST_value"]
        for image_id, image_data in subject_data.items():
            if image_id == "MST_value":
                continue
            
            # Extract sclera and CSEC cheek colors (use 0, 0, 0, if missing)
            sclera_left = image_data.get("sclera_RGB", {}).get("left_cheek_color") or [0, 0, 0]
            sclera_right = image_data.get("sclera_RGB", {}).get("right_cheek_color") or [0, 0, 0]
            csec_left = image_data.get("CSEC_cheek_colors", {}).get("left_cheek_color") or [0, 0, 0]
            csec_right = image_data.get("CSEC_cheek_colors", {}).get("right_cheek_color") or [0, 0, 0]
            
            # Extract FIQA exposure (use default if missing)
            fiqa_exposure = image_data.get("FIQA_exposure", {}).get("Prob_Class_0")
            if fiqa_exposure is None:
                fiqa_exposure = 0  # Default to 0 if missing
                        
            # Combine features
            combined_features = sclera_left + sclera_right + csec_left + csec_right + [fiqa_exposure]
            features.append(combined_features)
            targets.append(mst_value)
    
    return np.array(features), np.array(targets)


def save_results(output_file, model_name, accuracy, report):
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
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
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
            if image_name != "MST_value":  # Skip MST_value entry
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