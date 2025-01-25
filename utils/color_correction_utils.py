import numpy as np
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from scipy.ndimage import uniform_filter


def rgb_to_xyY(rgb):

    """
    Convert RGB values to CIE 1931 xyY color space.
    
    Args:
        rgb (numpy.ndarray): Input RGB values (normalized to [0, 1]).
        
    Returns:
        numpy.ndarray: Converted xyY values.
    """

    # Standard RGB to XYZ conversion matrix (assuming RGB is in [0, 1] range)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])


    xyz = np.dot(rgb, M.T)

    # Normalize XYZ to get xy
    x = xyz[0] / np.sum(xyz)
    y = xyz[1] / np.sum(xyz)
    
    Y = xyz[1]

    return np.array([x, y, Y])



def otsu_segmentation(rgb_image):
    """ Segment the input RGB image using Otsu's thresholding.

    Args:
        rgb_image (numpy.ndarray): Input RGB image.

    Returns:
        numpy.ndarray: Segmented image using Otsu's thresholding.
    """
    # Convert the RGB image to grayscale
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    
    _, thresholded = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresholded


def rgb_to_xyY(rgb):
    """ Convert RGB values to CIE 1931 xyY color space.

    Args:
        rgb (numpy.ndarray): Input RGB values.

    Returns:
        numpy.ndarray: Converted xyY values.
    """

    rgb = np.array(rgb, dtype=np.float32) / 255.0

    # Convert RGB to XYZ (D65 illuminant assumption)
    X = 0.4124564 * rgb[0] + 0.3575761 * rgb[1] + 0.1804375 * rgb[2]
    Y = 0.2126729 * rgb[0] + 0.7151522 * rgb[1] + 0.0721750 * rgb[2]
    Z = 0.0193339 * rgb[0] + 0.1191920 * rgb[1] + 0.9503041 * rgb[2]
    
    total = X + Y + Z
    x = X / total
    y = Y / total

    return (x, y, Y)

def von_kries_correction(image, illuminant_ratio):
    """ Apply von Kries chromatic adaptation to the input image.

    Args:
        image (numpy.ndarray): Input image in BGR format.
        illuminant_ratio (tuple): Illuminant ratio for chromatic adaptation.

    Returns:
        numpy.ndarray: Corrected image in BGR format.
    """

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = np.float32(image_rgb)

    # Apply the von Kries chromatic adaptation
    corrected_image = image_rgb * [illuminant_ratio[0], illuminant_ratio[1], 1]
    corrected_image = np.clip(corrected_image, 0, 255)

    corrected_image_bgr = cv2.cvtColor(np.uint8(corrected_image), cv2.COLOR_RGB2BGR)

    return corrected_image_bgr

def apply_otsu_thresholding(values):
    """ Apply Otsu's thresholding to the input values.

    Args:
        values (numpy.ndarray): Input

    Returns:
        numpy.ndarray: Thresholded image using Otsu's method.
    """

    # Convert the RGB values to grayscale
    gray_values = np.dot(values, [0.2989, 0.587, 0.114]) 
    
    _, thresholded = cv2.threshold(gray_values.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresholded

def create_eye_mask_otsu(locations, values, height, width):
    """ Create a mask for the eye based on the Otsu thresholded values.

    Args:
        locations (list): List of pixel locations.
        values (numpy.ndarray): RGB values.
        height (int): Image height.
        width (int): Image width.

    Returns:
        numpy.ndarray: Mask for the eye based on the Otsu thresholded values
    """

    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Apply Otsu's thresholding
    thresholded_mask = apply_otsu_thresholding(values)
    
    # Create a mask for the eye
    for i, (loc, threshold) in enumerate(zip(locations, thresholded_mask)):
        x, y = loc
        if threshold == 255 and 0 <= x < width and 0 <= y < height:
            mask[y, x] = 255 
    
    return mask


def plot_sclera(image, sclera_coords, masked_image, l_avg_rgb, r_avg_rgb, title):
    """ Plot the cropped sclera region along with the average RGB values for the left and right sclera.

    Args:
        image (numpy.ndarray): Original input image.
        sclera_coords (numpy.ndarray): Coordinates of the sclera region.
        masked_image (numpy.ndarray): Image with the sclera region masked.
        l_avg_rgb (numpy.ndarray): Average RGB value for the left sclera.
        r_avg_rgb (numpy.ndarray): Average RGB value for the right sclera.
        title (str): Title for the plot
    """

    y_min, x_min = sclera_coords.min(axis=0)
    y_max, x_max = sclera_coords.max(axis=0)
    #x_max = x_max + 90
    # Add a margin to the bounding box
    margin = 5
    y_min = max(y_min - margin, 0)
    x_min = max(x_min - margin, 0)
    y_max = min(y_max + margin, image.shape[0])
    x_max = min(x_max + margin, image.shape[1])

    cropped_region = masked_image[y_min:y_max, x_min:x_max]

    # plot the cropped region
    plt.figure(figsize=(15, 7))
    # plt.title(title, fontsize=15)
    plt.imshow(cv2.cvtColor(cropped_region, cv2.COLOR_BGR2RGB))
    plt.axis('off')


    cropped_original_eyes = image[y_min-20:y_max+20, x_min-5:x_max]

    plt.figure(figsize=(15, 7))
    # plt.title(title, fontsize=15)
    plt.imshow(cropped_original_eyes)
    plt.axis('off')

    # save the figure
    # plt.savefig(title)

    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    
    ax[0].imshow(np.full((100, 100, 3), r_avg_rgb, dtype=np.uint8))
    ax[0].set_title('Right Sclera, RGB: ' + str(r_avg_rgb.astype(int)), fontsize=10)
    ax[0].axis('off')

    ax[1].imshow(np.full((100, 100, 3), l_avg_rgb, dtype=np.uint8))
    ax[1].set_title('Left Sclera, RGB: ' + str(l_avg_rgb.astype(int)), fontsize=10)
    ax[1].axis('off')

    plt.show()
    


def masked_moving_average(image, mask, patch_size=5):
    """ Compute the moving average of the image using the mask.

    Args:
        image (numpy.ndarray): Input image.
        mask (numpy.ndarray): Binary mask.
        patch_size (int): Size of the patch for the moving average.

    Returns:
        numpy.ndarray: Moving average of the image.
    """

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = mask.astype(float)  
    mask = mask[:, :, np.newaxis] 
    
    # Compute moving sum for image and mask
    moving_sum_image = uniform_filter(image * mask, size=(patch_size, patch_size, 1))
    moving_sum_mask = uniform_filter(mask, size=(patch_size, patch_size, 1))
    
    moving_avg = np.divide(moving_sum_image, moving_sum_mask, where=moving_sum_mask > 0)
    moving_avg[np.squeeze(moving_sum_mask, axis=-1) == 0] = 0
    
    # Calculate the valid mask sum
    valid_mask_sum = np.sum(moving_sum_mask > 0)  
    avg_rgb = np.sum(moving_avg, axis=(0, 1)) / valid_mask_sum
    
    return avg_rgb

def calculate_patch_avg_color(img, mask, coords, window_size=10, stride=5):
    """
    Calculate the average RGB color using a sliding window over the specified coordinates,
    and plot the bounding box and extracted patch colors.

    Args:
        img (ndarray): The image array.
        mask (ndarray): Binary mask array (same dimensions as img).
        coords (tuple): The coordinates of the patch (x_min, x_max, y_min, y_max).
        window_size (int): The size of the sliding window (default is 10).
        stride (int): The stride of the sliding window (default is 5).

    Returns:
        list: List of mean colors for each valid window.
    """
    x_min, x_max, y_min, y_max = (int(coord) for coord in coords)
    patch_colors = []

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = np.array(img)
    img_with_bbox = img.copy()
    img_with_windows = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB).copy()

    # Draw the bounding box on the image
    cv2.rectangle(
        img_with_bbox, 
        (x_min, y_min), 
        (x_max, y_max), 
        (255, 0, 0),  # Red bounding box
        2
    )

    # Slide the window over the specified coordinates
    for y in range(y_min, y_max - window_size + 1, stride):
        for x in range(x_min, x_max - window_size + 1, stride):
            # Extract the window
            window = img[y:y + window_size, x:x + window_size]
            mask_window = mask[y:y + window_size, x:x + window_size]
            
            valid_pixels_canvas = np.zeros_like(window)
            # Select only valid (unmasked) pixels
            valid_pixels = window[mask_window == 255]
            valid_pixels = valid_pixels[np.any(valid_pixels != [0, 0, 0], axis=1)]

            # print(len(valid_pixels))
            if len(valid_pixels):

                mean_color = np.mean(valid_pixels.reshape(-1, 3), axis=0).astype(int)
                patch_colors.append(mean_color)

                # valid_indices = np.where(mask_window == 255)
                # valid_pixels_canvas[valid_indices[0], valid_indices[1]] = valid_pixels

                # for i, pixel in enumerate(valid_pixels):
                #     print(f"Pixel {i + 1}: {pixel}")

                # fig, ax = plt.subplots(1, 2, figsize=(12, 5))

                # # Plot the valid pixels canvas
                # ax[0].imshow(valid_pixels_canvas)
                # ax[0].set_title(f"Valid Pixels in Window ({x}, {y})")
                # ax[0].axis("off")

                # # Plot the mean color as a solid color block
                # ax[1].imshow([[mean_color / 255]])  # Normalize color for display (between 0 and 1)
                # ax[1].set_title(f"Mean Color\nRGB: {mean_color}")
                # ax[1].axis("off")

                # # Display the plot
                # plt.tight_layout()
                # plt.show()

                cv2.rectangle(img_with_windows, (x, y), (x + window_size, y + window_size), (0, 255, 0), 1)
            else:
                cv2.rectangle(img_with_windows, (x, y), (x + window_size, y + window_size), (255, 0, 0), 1)

    # Display the accumulated patches
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

    # # Display the image with all windows highlighted
    # plt.figure(figsize=(20, 20))
    # plt.imshow(img_with_windows)
    # plt.title("Image with Highlighted Windows")
    # plt.axis('off')
    # plt.show()

    # Plot the results
    # fig, ax = plt.subplots(1, len(patch_colors) + 1, figsize=(20, 5))

    # if not isinstance(ax, np.ndarray):
    #     ax = [ax]
        
    # print("patch_colors", patch_colors)
    # print(len(patch_colors))
    # # Plot the image with bounding box

    # # Plot the image with bounding box
    # ax[0].imshow(cv2.cvtColor(img_with_bbox, cv2.COLOR_BGR2RGB))
    # ax[0].set_title("Image with Bounding Box")
    # ax[0].axis('off')

    # # Plot the patch colors
    # for i, color in enumerate(patch_colors):
    #     ax[i + 1].imshow([[color / 255]])  # Normalize to [0, 1] for display
    #     # ax[i + 1].set_title(f"Patch {i + 1}\nRGB: {color}")
    #     ax[i + 1].axis('off')

    # plt.tight_layout()
    # plt.show()

    # Return the list of mean patch colors

    if patch_colors:
        avg_color = np.mean(patch_colors, axis=0).astype(int)
    else:
        avg_color = None

    return avg_color

import numpy as np

def calculate_coords(mask):
    """ Calculate the bounding box coordinates for the mask.

    Args:
        mask (numpy.ndarray): Binary mask.

    Returns:
        tuple: Bounding box coordinates (x_min, x_max, y_min
    """

    # Find the nonzero coordinates in the mask
    nonzero_y, nonzero_x = np.nonzero(mask) 
    
    x_min = np.min(nonzero_x)
    x_max = np.max(nonzero_x)
    y_min = np.min(nonzero_y)
    y_max = np.max(nonzero_y)
    
    return x_min, x_max, y_min, y_max


def adjust_exposure(image, exposure_factor):
    """ Adjust the exposure of the input image by multiplying it with the exposure factor.

    Args:
        image (numpy.ndarray): Input image.
        exposure_factor (float): Exposure factor to adjust the image.

    Returns:
        numpy.ndarray: Adjusted image
    """

    image = np.float32(image) 
    adjusted_image = image * exposure_factor

    return np.clip(adjusted_image, 0, 255).astype(np.uint8)

def plot_images(image, image_corrected_exposure, corrected_image, corrected_exposure_color_image):
    """ Plot the original and corrected images side by side.

    Args:
        image (numpy.ndarray): Original input image.
        image_corrected_exposure (numpy.ndarray): Image with exposure correction.
        corrected_image (numpy.ndarray): Image with color correction.
        corrected_exposure_color_image (numpy.ndarray): Image with exposure and color
    """


    # Plot original and corrected images side by side
    plt.figure(figsize=(15, 8))

    # Original Image
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(image_corrected_exposure)
    plt.title('Exposure Correction')
    plt.axis('off')

    # Corrected Image
    plt.subplot(1, 4, 3)
    plt.imshow(corrected_image)
    plt.title('Color Correction')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(corrected_exposure_color_image)
    plt.title('Exposure and Color Correction')
    plt.axis('off')


    plt.show()

def plot_gray_scale_histograms(rgb_array, title):
    """ Plot the grayscale histogram of the input RGB array.

    Args:
        rgb_array (numpy.ndarray): Input RGB array.
        title (str): Title for the plot.
    """
    # Convert to grayscale using the formula
    grayscale_values = 0.2989 * rgb_array[:, 0] + 0.5870 * rgb_array[:, 1] + 0.1140 * rgb_array[:, 2]

    # Plot the grayscale histogram
    plt.hist(grayscale_values, bins=100, color='gray', edgecolor='gray')
    plt.ylim(0, 10)

    # plt.title(title)
    plt.xlabel('Gray Intensity Value')
    plt.ylabel('Frequency')
    plt.savefig(title)
    plt.show()
