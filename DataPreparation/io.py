import os
import random
import pathlib
import cv2
import pandas as pd
import numpy as np
from natsort import natsorted

def load_image_and_samples(img_dir: str, output_dir: str, img_name: str):
    """
    Load a grayscale image and extract landmark coordinates from all sample files.

    Parameters:
      img_dir (str): Directory containing the input image.
      output_dir (str): Directory containing the sample output files.
      img_name (str): Name of the image file (e.g., 'image.png').

    Returns:
      df (pd.DataFrame): DataFrame with columns 'Node', 'Sample', 'X', 'Y'.
      img (np.ndarray): Loaded grayscale image as a NumPy array.
    """
    img_path = os.path.join(img_dir, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {img_path}")

    img_name = pathlib.PosixPath(img_name)
    base_path = str(img_name.parent)
    base_name = img_name.stem + '_'
    
    if base_path != '.':
        output_dir += '/' + base_path

    sample_files = [f for f in os.listdir(output_dir) if f.startswith(base_name) and f.endswith('.txt')]
    if not sample_files:
        raise FileNotFoundError(f"No sample files found for {base_name} in {output_dir}")

    sample_files = natsorted(sample_files)
    data = []
    for sample_idx, sample_file in enumerate(sample_files, 1):
        output_path = os.path.join(output_dir, sample_file)
        x_coords = []
        y_coords = []
        with open(output_path, 'r') as file:
            for line in file:
                x, y = map(float, line.strip().split())
                x_coords.append(x)
                y_coords.append(y)
        for i in range(len(x_coords)):
            data.append({
                'Node': i + 1,
                'Sample': sample_idx,
                'X': x_coords[i],
                'Y': y_coords[i]
            })
    df = pd.DataFrame(data)
    return df, img

def extract_landmarks(file_name: str, split: bool = True) -> dict:
    """
    Extract ground-truth landmarks from .npy files in the Annotations directory.

    Parameters:
      file_name (str): The image file name.
      split (bool): If True, removes the last underscore segment from file_name.

    Returns:
      dict: Dictionary with keys 'RL', 'LL', 'H' containing the corresponding numpy arrays or None.
    """
    landmarks_dir = 'Annotations'
    landmarks_dict = {}
    
    if split:
        file_name = '_'.join(file_name.split('_')[:-1])
    
    for key in ['RL', 'LL', 'H']:
        file_path = os.path.join(landmarks_dir, key, file_name + '.npy')
        if os.path.exists(file_path):
            landmarks_dict[key] = np.load(file_path)
        else:
            landmarks_dict[key] = None  
    return landmarks_dict

def sample_landmarks(landmark_dict, organ, n_samples=1):
    if landmark_dict.get(organ) is None:
        return np.empty((0,2))
    else:
        landmarks = landmark_dict[organ]
        n_samples = min(n_samples, len(landmarks))
        
        indices = np.random.choice(len(landmarks), n_samples, replace=False)

        return landmarks[indices]