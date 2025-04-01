import os
import random
import cv2
import pandas as pd
import numpy as np

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

    base_name = img_name.split('.png')[0]
    sample_files = [f for f in os.listdir(output_dir) if f.startswith(base_name) and f.endswith('.txt')]
    if not sample_files:
        raise FileNotFoundError(f"No sample files found for {base_name} in {output_dir}")
    sample_files.sort(key=lambda x: int(x.split('_')[-1].split('.txt')[0]))

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
    
def sample_images(img_dir: str, num_samples: int = 15) -> list:
    all_images = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    return random.sample(all_images, min(num_samples, len(all_images)))

def process_images(img_dir: str, output_subdir: str, process_fn, severity_levels: list, **kwargs):
    output_dir = os.path.join(img_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    selected_images = sample_images(img_dir)
    
    for img_name in selected_images:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_base, ext = os.path.splitext(img_name)
        
        for severity in severity_levels:
            processed_img = process_fn(img, severity, **kwargs)
            output_name = f"{img_base}_{severity}{ext}"
            output_path = os.path.join(output_dir, output_name)
            cv2.imwrite(output_path, processed_img)