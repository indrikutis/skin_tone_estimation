import numpy as np
from skimage import color
from skimage.io import imread



def rgb_to_xyy(rgb):
    """ Convert RGB color to xyY color space.

    Args:
        rgb (np.array): RGB color values in the range [0, 255].

    Returns:
        np.array: xyY color values.
    """
    rgb = rgb / 255.0 
    xyz = color.rgb2xyz(rgb)
    
    # Convert XYZ to xyY
    x = xyz[0] / (xyz[0] + xyz[1] + xyz[2])
    y = xyz[1] / (xyz[0] + xyz[1] + xyz[2])
    Y = xyz[1]  
    
    return np.array([x, y, Y])

def get_average_sclera_color(image_path, l_locations, r_locations):

    """ Get the average sclera color from the image.

    Args:
        image_path (str): Path to the image.
        l_locations (list): List of pixel coordinates for the left sclera.
        r_locations (list): List of pixel coordinates for the right sclera.

    Returns:
        np.array: Average sclera color in xyY color space.
    """

    image = imread(image_path)

    # Ensure l_locations and r_locations are NumPy arrays for correct indexing
    l_locations = np.array(l_locations) if l_locations else None
    r_locations = np.array(r_locations) if r_locations else None

    l_values = []
    r_values = []

    # Get the pixel values for the left sclera, if present
    if l_locations is not None and l_locations.size > 0:
        h, w = image.shape[:2] 
        valid_l_locations = (l_locations[:, 0] < w) & (l_locations[:, 1] < h)  
        l_values = image[l_locations[valid_l_locations, 1], l_locations[valid_l_locations, 0]] 

    # Get the pixel values for the right sclera, if present
    if r_locations is not None and r_locations.size > 0:
        valid_r_locations = (r_locations[:, 0] < w) & (r_locations[:, 1] < h)  
        r_values = image[r_locations[valid_r_locations, 1], r_locations[valid_r_locations, 0]]  

    # Calculate the average sclera color
    if len(l_values) > 0 and len(r_values) > 0:
        avg_left = np.mean(l_values, axis=0)
        avg_right = np.mean(r_values, axis=0)
        avg_sclera_color = (avg_left + avg_right) / 2
    elif len(l_values) > 0:  
        avg_sclera_color = np.mean(l_values, axis=0)
    elif len(r_values) > 0: 
        avg_sclera_color = np.mean(r_values, axis=0)
    else:
        return None

    # Convert average RGB to xyY
    avg_xyY = rgb_to_xyy(avg_sclera_color)

    return avg_xyY


def convert_to_native_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to lists
    elif isinstance(obj, np.int64):
        return int(obj)  # Convert np.int64 to regular int
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}  # Recursively convert dict
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]  # Recursively convert list
    else:
        return obj  # Return other types as they are

def clip_coordinates(locations, height, width):
    # Ensure x and y coordinates are within the bounds of the image
    clipped_locations = []
    for loc in locations:
        x = min(max(loc[0], 0), width - 1) 
        y = min(max(loc[1], 0), height - 1)  
        clipped_locations.append((x, y))
    return clipped_locations